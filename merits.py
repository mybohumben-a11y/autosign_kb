import os
import sys
import datetime
import traceback

# 라이브러리 설치 확인 및 임포트
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF (fitz) 라이브러리가 설치되어 있지 않습니다.")
    print("명령 프롬프트(cmd)에서 'pip install pymupdf'를 실행해 주세요.")
    input("\n엔터를 누르면 종료됩니다...")
    sys.exit()

def get_base_path():
    """EXE 실행 파일로 빌드 시 경로 오류를 방지하는 함수"""
    if getattr(sys, 'frozen', False):
        # .exe로 실행 중일 때
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        # .py로 실행 중일 때
        return os.path.dirname(os.path.abspath(__file__))

def process_pdfs():
    base_path = get_base_path()
    font_path = "UhBee Creator.ttf"
    
    # 폰트 존재 여부 확인
    if not os.path.exists(font_path):
        print(f"폰트 파일을 찾을 수 없습니다: {font_path}")
        return

    # 오늘 날짜 정보 (2자리)
    now = datetime.datetime.now()
    year_str = now.strftime("%y")
    month_str = now.strftime("%m")
    day_str = now.strftime("%d")
    timestamp = now.strftime("%H%M%S")

    # 대상 파일 리스트 필터링
    pdf_files = [f for f in os.listdir(base_path) 
                 if f.lower().endswith(".pdf") and "날짜보정완료" not in f]

    if not pdf_files:
        print("작업할 PDF 파일이 없습니다. (이미 완료되었거나 파일이 존재하지 않음)")
        return

    print(f"총 {len(pdf_files)}개의 파일을 발견했습니다. 작업을 시작합니다...\n")

    for filename in pdf_files:
        doc = None
        try:
            print(f"처리 중: {filename}")
            file_path = os.path.join(base_path, filename)
            doc = fitz.open(file_path)
            
            # 기준 이름 좌표 저장을 위한 변수
            ref_name_coords = None
            target_name = ""

            # ---------------------------------------------------------
            # 1. 2페이지에서 "동의자" 찾기 및 이름 기준 좌표 획득
            # ---------------------------------------------------------
            if len(doc) >= 2:
                page2 = doc[1]  # 0부터 시작하므로 1이 2페이지
                words = page2.get_text("words")
                
                for i, w in enumerate(words):
                    # w[4]는 텍스트 내용임
                    if "동의자" in w[4]:
                        # 바로 뒤의 단어를 이름으로 간주
                        if i + 1 < len(words):
                            name_word = words[i+1]
                            target_name = name_word[4]
                            # 기준점: x0(왼쪽), y0(위쪽) 저장
                            ref_name_coords = (name_word[0], name_word[1])
                            break
            
            # ---------------------------------------------------------
            # 2. 모든 페이지: "동의함" 찾아 V 표시 (크기 21)
            # ---------------------------------------------------------
            for page in doc:
                found_agrees = page.search_for("동의함")
                for rect in found_agrees:
                    # 위치: x0 + 8, y1 + 18
                    v_x = rect.x0 + 8
                    v_y = rect.y1 + 18
                    page.insert_text((v_x, v_y), "V", fontname="kor", fontfile=font_path, fontsize=21)

            # ---------------------------------------------------------
            # 3. 2페이지: 이름 및 날짜 쓰기 (크기 11)
            # ---------------------------------------------------------
            if len(doc) >= 2:
                page2 = doc[1]
                
                # 이름 쓰기 (기억해둔 좌표 기준)
                if ref_name_coords and target_name:
                    nx, ny = ref_name_coords
                    page2.insert_text((nx + 85, ny + 10), target_name, fontname="kor", fontfile=font_path, fontsize=11)

                # 날짜 쓰기 ("동의일자" 찾기)
                date_labels = page2.search_for("동의일자")
                if date_labels:
                    # 첫 번째 검색된 영역 기준
                    rect = date_labels[0]
                    x1, y1 = rect.x1, rect.y1
                    
                    # 년, 월, 일 기입
                    page2.insert_text((x1 + 80, y1 - 2), year_str, fontname="kor", fontfile=font_path, fontsize=11)
                    page2.insert_text((x1 + 115, y1 - 2), month_str, fontname="kor", fontfile=font_path, fontsize=11)
                    page2.insert_text((x1 + 155, y1 - 2), day_str, fontname="kor", fontfile=font_path, fontsize=11)

            # ---------------------------------------------------------
            # 파일 저장
            # ---------------------------------------------------------
            name_body = os.path.splitext(filename)[0]
            new_filename = f"{name_body}_날짜보정완료_{timestamp}.pdf"
            save_path = os.path.join(base_path, new_filename)
            
            doc.save(save_path)
            print(f"저장 완료: {new_filename}")

        except Exception:
            print(f"!!! {filename} 처리 중 오류 발생 !!!")
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
        print("\n" + "="*50)
        print("모든 작업이 종료되었습니다.")
        input("엔터를 누르면 종료됩니다...")