import os
import sys
import time
import traceback
from datetime import datetime
import fitz  # PyMuPDF
import cv2
import numpy as np

def process_pdf_workflow():
    """
    지시서에 따른 PDF 자동 보정 및 서명 기입 워크플로우를 수행합니다.
    """
    # 1. 환경 설정 및 리소스 확인
    font_path = "C:/Windows/Fonts/malgun.ttf"
    template_path = "NH_image.png"
    
    # 필수 자원 체크
    if not os.path.exists(template_path):
        print(f"[오류] 템플릿 이미지({template_path})가 없습니다.")
        return
    if not os.path.exists(font_path):
        print(f"[경고] {font_path} 경로에 폰트가 없어 기본 폰트를 사용하거나 오류가 발생할 수 있습니다.")

    # 실행 경로 설정 (일반 실행 및 EXE 빌드 대응)
    if hasattr(sys, '_MEIPASS'):
        current_dir = os.path.dirname(sys.executable)
    else:
        current_dir = os.getcwd()

    # 대상 파일 식별 (제외 대상 필터링)
    pdf_files = [f for f in os.listdir(current_dir) 
                 if f.lower().endswith('.pdf') and "날짜보정완료" not in f]

    if not pdf_files:
        print("[-] 처리할 대상 PDF 파일이 없습니다.")
        return

    # 템플릿 로드 (그레이스케일)
    template = cv2.imread(template_path, 0)
    
    # 오늘 날짜의 연도 (YYYY)
    current_year = str(datetime.now().year)
    print(f"[*] 분석 기준 연도: {current_year}")

    for file_name in pdf_files:
        doc = None
        try:
            print(f"\n--- [작업 시작: {file_name}] ---")
            pdf_path = os.path.join(current_dir, file_name)
            doc = fitz.open(pdf_path)
            
            # --- 2. 데이터 추출 (1페이지) ---
            page1 = doc[0]
            words_p1 = page1.get_text("words")
            target_name = "고객"
            
            for i, w in enumerate(words_p1):
                # 단어 리스트 중 현재 연도(YYYY)가 포함된 텍스트 탐색
                if current_year in w[4]:
                    if i > 0:
                        target_name = words_p1[i-1][4]
                        print(f"  ▶ 추출된 고객명: {target_name}")
                    break
            
            # --- 3~4. 시각 분석 및 기입 (전 페이지 대상) ---
            for page_index, page in enumerate(doc):
                # 한글 폰트 삽입
                page.insert_font(fontname="kor", fontfile=font_path)
                
                # [이미지 변환] 고해상도 분석용 (zoom=2)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                found_rects = []
                threshold = 0.7
                
                # [멀티 스케일 매칭] 0.5배 ~ 1.5배 사이 탐색
                for scale in np.linspace(0.5, 1.5, 11):
                    resized_t = cv2.resize(template, None, fx=scale, fy=scale)
                    if resized_t.shape[0] > img_gray.shape[0] or resized_t.shape[1] > img_gray.shape[1]:
                        continue
                        
                    res = cv2.matchTemplate(img_gray, resized_t, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)
                    
                    t_w, t_h = resized_t.shape[::-1]
                    for pt in zip(*loc[::-1]):
                        pdf_x, pdf_y = pt[0] / zoom, pt[1] / zoom
                        pdf_w, pdf_h = t_w / zoom, t_h / zoom
                        
                        # 중복 좌표 제거 (15pt 이내)
                        if not any(abs(pdf_x - ex) < 15 and abs(pdf_y - ey) < 15 for ex, ey, _, _ in found_rects):
                            found_rects.append((pdf_x, pdf_y, pdf_w, pdf_h))

                # [V 마킹 기입] 탐지된 좌표 우측 이동 반영
                for (x, y, w, h) in found_rects:
                    page.insert_text((x + 92, y + h + 2), "V", 
                                     fontname="kor", fontsize=18, color=(0, 0, 0))

                # [성명 기입] PDF 전체 페이지에서 target_name 위치를 찾아 서명란에 기입
                p_words = page.get_text("words")
                for w in p_words:
                    if target_name in w[4]:
                        # 지시 사항에 따라 (x + 20, y50) 좌표에 기입
                        page.insert_text((w[0] + 13, w[1]+29), target_name, 
                                         fontname="kor", fontsize=11, color=(0, 0, 0))
                        # 중복 기입을 피하려면 한 페이지에 한 번만 찾고 break 할 수 있음
                        # 여기서는 발견될 때마다 기입하도록 설정됨
                
                print(f"  - {page_index+1}p 분석 및 기입 완료 (V 탐지: {len(found_rects)}개)")

            # --- 5. 파일 저장 및 최적화 ---
            timestamp = time.strftime('%H%M%S')
            save_name = f"{os.path.splitext(file_name)[0]}_날짜보정완료_{timestamp}.pdf"
            output_path = os.path.join(current_dir, save_name)
            
            doc.save(
                output_path,
                garbage=4,     # 미사용 객체 제거
                deflate=True,  # 스트림 압축
                clean=True     # 문서 구조 정리
            )
            print(f"[성공] 저장 완료: {save_name}")

        except Exception:
            print(f"[실패] {file_name} 처리 중 오류 발생")
            traceback.print_exc()
        finally:
            if doc:
                doc.close()

if __name__ == "__main__":
    try:
        # 수정됨: 정의된 함수 이름과 호출 이름을 일치시킴
        process_pdf_workflow()
    finally:
        print("\n" + "="*40)
        input("작업이 종료되었습니다. 엔터를 누르세요...")