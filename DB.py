import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta

# 1. PyMuPDF (fitz) 라이브러리 체크 및 임포트
try:
    import fitz  # PyMuPDF
except ImportError:
    print("오류: PyMuPDF(fitz) 라이브러리가 설치되어 있지 않습니다.")
    print("명령 프롬프트(cmd)에서 'pip install pymupdf'를 실행하여 설치해 주세요.")
    input("\n엔터를 누르면 종료됩니다...")
    sys.exit()

def get_base_path():
    """실행 파일(.exe) 또는 스크립트가 위치한 경로 반환"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))

def process_pdfs():
    base_path = get_base_path()
    # 작업 디렉토리 변경
    os.chdir(base_path)
    
    # 윈도우 맑은 고딕 폰트 경로
    font_path = "UhBee Creator.ttf"
    if not os.path.exists(font_path):
        print(f"경고: {font_path} 경로에 폰트가 없습니다. 시스템 기본 설정을 확인하세요.")

    # 파일 목록 가져오기 (PDF만, '날짜보정완료' 제외)
    pdf_files = [f for f in os.listdir(base_path) 
                 if f.lower().endswith('.pdf') and "날짜보정완료" not in f]

    if not pdf_files:
        print("처리할 대상 PDF 파일이 폴더에 없습니다.")
        return

    print(f"총 {len(pdf_files)}개의 파일을 찾았습니다.")

    for filename in pdf_files:
        doc = None
        try:
            print(f"--- [{filename}] 처리 시작 ---")
            doc = fitz.open(filename)
            
            # --- 1단계: 이름 찾기 (1페이지) ---
            client_name = "고객" 
            page1 = doc[0]
            words1 = page1.get_text("words") 
            
            for i, w in enumerate(words1):
                text_content = w[4].strip()
                if "고객님" in text_content:
                    # Case 1: "정광훈 고객님"처럼 한 단어에 이름이 붙어 있는 경우
                    if text_content != "고객님":
                        client_name = text_content.replace("고객님", "").strip()
                    # Case 2: "정광훈" / "고객님"으로 분리되어 있어 앞 단어를 가져와야 하는 경우
                    elif i > 0:
                        client_name = words1[i-1][4].strip()
                    
                    print(f"추출된 이름: {client_name}")
                    break

            # --- 2단계: 기준점 "구분" 위치 찾기 (2페이지) ---
            ref_x0, ref_y0 = None, None
            if len(doc) >= 2:
                page2 = doc[1]
                words2 = page2.get_text("words")
                for w in words2:
                    if "구분" in w[4]:
                        ref_x0, ref_y0 = w[0], w[1] # 왼쪽(x0), 위쪽(y0)
                        print(f"'구분' 기준 좌표 확보: ({ref_x0}, {ref_y0})")
                        break
            
            # --- 3단계: 내용 수정 (모든 페이지) ---
            for page_idx, page in enumerate(doc):
                page.insert_font(fontname="kor", fontfile=font_path)
                
                # V자 체크: "동의함" 우측 +40 (DB.py 기준 좌표 및 크기 반영)
                v_targets = page.search_for("동의함")
                for rect in v_targets:
                    # y좌표는 약간 위로(-2), 폰트 크기는 15
                    v_point = fitz.Point(rect.x0 + 40, rect.y1 - 2)
                    page.insert_text(v_point, "V", fontname="kor", fontsize=15, color=(0, 0, 0))
                
                # 이름 기입: 2페이지 (DB.py 기준 좌표 반영)
                if page_idx == 1 and ref_x0 is not None:
                    # 첫 번째 위치: x0 + 45, y0 + 27
                    p1 = fitz.Point(ref_x0 + 45, ref_y0 + 27)
                    page.insert_text(p1, client_name, fontname="kor", fontsize=11)
                    
                    # 두 번째 위치: x0 + 45, y0 + 60
                    p2 = fitz.Point(ref_x0 + 45, ref_y0 + 60)
                    page.insert_text(p2, client_name, fontname="kor", fontsize=11)

            # 새 파일명 생성 및 저장
            timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%H%M%S")
            name_only = os.path.splitext(filename)[0]
            new_filename = f"{name_only}_날짜보정완료_{timestamp}.pdf"
            
            doc.save(new_filename)
            print(f"성공: {new_filename} 저장 완료")

        except Exception:
            print(f"에러 발생: {filename} 처리 중 오류가 발생했습니다.")
            traceback.print_exc()
        finally:
            if doc:
                doc.close()

if __name__ == "__main__":
    try:
        process_pdfs()
    except Exception:
        traceback.print_exc()
    finally:
        print("\n" + "="*40)
        print("모든 작업이 종료되었습니다.")
        input("엔터를 누르면 종료됩니다...")