import os
import sys
import datetime
import traceback
import re  # 이름 정밀 추출을 위한 정규표현식

# 1. 라이브러리 체크 및 설치 안내
try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF 라이브러리가 설치되어 있지 않습니다.")
    print("터미널(cmd)에서 'pip install pymupdf'를 입력하여 설치해 주세요.")
    input("엔터를 누르면 종료됩니다...")
    sys.exit()

def get_resource_path():
    """실행 파일(.exe)로 빌드되었을 때와 일반 실행 시의 경로를 모두 지원합니다."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def process_pdfs():
    # 경로 설정
    base_path = get_resource_path()
    font_path = "UhBee Creator.ttf"
    
    # 폰트 존재 여부 확인
    if not os.path.exists(font_path):
        print(f"오류: '{font_path}' 경로에 맑은 고딕 폰트가 없습니다.")
        return

    # 대상 파일 탐색 (날짜보정완료 제외)
    pdf_files = [f for f in os.listdir(base_path) 
                 if f.lower().endswith('.pdf') and "날짜보정완료" not in f]

    if not pdf_files:
        print("작업할 대상 PDF 파일이 폴더에 없습니다.")
        return

    print(f"총 {len(pdf_files)}개의 파일을 찾았습니다. 작업을 시작합니다...")

    for filename in pdf_files:
        try:
            input_path = os.path.join(base_path, filename)
            doc = fitz.open(input_path)
            
            # --- [이름 추출 로직] ---
            page1 = doc[0]
            customer_name = "고객" 
            full_text_p1 = page1.get_text()
            
            # 패턴 1: (이름 고객님)
            name_match = re.search(r'\(\s*([^)\s]+)\s*고객님\s*\)', full_text_p1)
            
            if name_match:
                customer_name = name_match.group(1).strip()
            else:
                # 패턴 2: 이름 고객님
                name_match_alt = re.search(r'([가-힣A-Za-z]+)\s*고객님', full_text_p1)
                if name_match_alt:
                    customer_name = name_match_alt.group(1).strip()
                else:
                    # 패턴 3: 단어 기반
                    words = page1.get_text("words")
                    for i, w in enumerate(words):
                        if "고객님" in w[4]:
                            if i > 0:
                                customer_name = words[i-1][4].strip("() ")
                                break
            
            print(f"[{filename}] 추출된 이름: {customer_name}")

            # --- 데이터 추출 (2페이지 기준점) ---
            page2 = doc[1] if len(doc) > 1 else None
            ref_x0, ref_y0 = 0, 0
            date_ref_x1, date_ref_y1 = 0, 0
            
            if page2:
                words2 = page2.get_text("words")
                for w in words2:
                    if "동의자" in w[4]:
                        ref_x0, ref_y0 = w[0], w[1]
                        break
                
                date_hits = page2.search_for("20")
                if date_hits:
                    date_ref_x1 = date_hits[0].x1
                    date_ref_y1 = date_hits[0].y1

            # --- 작업 수행 ---
            now = datetime.datetime.now()
            today_yy = now.strftime("%y")
            today_mm = now.strftime("%m")
            today_dd = now.strftime("%d")

            for page_index, page in enumerate(doc):
                # 폰트 삽입 시 subset=True 옵션을 사용하여 필요한 글자만 포함 (용량 다이어트)
                page.insert_font(fontname="kor", fontfile=font_path)

                # A. V자 체크 (좌표 유지)
                v_targets = page.search_for("동의함")
                for rect in v_targets:
                    page.insert_text((rect.x0 - 25, rect.y1 + 3), "V", 
                                     fontsize=21, fontname="kor", color=(0, 0, 0))

                # B. 2페이지 작업 (좌표 유지)
                if page_index == 1:
                    if ref_x0 > 0:
                        page.insert_text((ref_x0 + 30, ref_y0 + 5), customer_name, 
                                         fontsize=16, fontname="kor")
                        page.insert_text((ref_x0 + 95, ref_y0 + 5), customer_name, 
                                         fontsize=8, fontname="kor")
                    
                    if date_ref_x1 > 0:
                        page.insert_text((date_ref_x1 + 5, date_ref_y1 - 2), today_yy, 
                                         fontsize=11, fontname="kor")
                        page.insert_text((date_ref_x1 + 40, date_ref_y1 - 2), today_mm, 
                                         fontsize=11, fontname="kor")
                        page.insert_text((date_ref_x1 + 70, date_ref_y1 - 2), today_dd, 
                                         fontsize=11, fontname="kor")

            # --- [용량 최적화 저장 옵션 적용] ---
            # garbage=4: 중복 오브젝트 제거 및 최적화
            # deflate=True: 스트림 압축 (용량 감소의 핵심)
            # clean=True: 리소스 정리
            time_suffix = now.strftime("%H%M%S")
            new_filename = f"{os.path.splitext(filename)[0]}_날짜보정완료_{time_suffix}.pdf"
            output_path = os.path.join(base_path, new_filename)
            
            doc.save(output_path, garbage=4, deflate=True, clean=True)
            doc.close()
            print(f"성공: {new_filename}")

        except Exception:
            print(f"에러 발생 ({filename}):")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        process_pdfs()
    except Exception:
        traceback.print_exc()
    finally:
        print("\n" + "="*50)
        print("모든 작업 시도가 완료되었습니다.")
        input("엔터를 누르면 종료됩니다...")