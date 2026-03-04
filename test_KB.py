import os
import sys
import time
import traceback
import fitz  # PyMuPDF
import cv2
import numpy as np

def process_pdf_multi_scale():
    # 1. 환경 설정
    font_path = "UhBee Creator.ttf"
    template_path = "image_3664f7.png"  # 템플릿 이미지 파일명
    
    if not os.path.exists(template_path):
        print(f"[오류] {template_path} 파일이 없습니다.")
        return

    # 실행 경로 설정 (EXE 빌드 시 내부 경로 대응)
    current_dir = os.path.dirname(sys.executable) if hasattr(sys, '_MEIPASS') else os.getcwd()
    pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf') and "날짜보정완료" not in f]

    # 템플릿 이미지 읽기 (그레이스케일)
    template = cv2.imread(template_path, 0)
    
    for file_name in pdf_files:
        doc = None
        try:
            print(f"\n--- 분석 시작: {file_name} ---")
            doc = fitz.open(os.path.join(current_dir, file_name))
            
            # [1단계] 고객명 추출 (1페이지)
            page1 = doc[0]
            words_p1 = page1.get_text("words")
            # "고객명" 텍스트 뒤에 오는 값을 이름으로 추출
            target_name = next((words_p1[i+2][4] for i, w in enumerate(words_p1) if "고객명" in w[4]), "고객")
            print(f"추출된 이름: {target_name}")

            for page_index, page in enumerate(doc):
                # 한글 폰트 삽입
                page.insert_font(fontname="kor", fontfile=font_path)
                
                # [2단계] PDF 페이지 이미지화 (분석용 고해상도 zoom=2)
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                # [3단계] 멀티 스케일 매칭 (0.5배 ~ 1.5배 사이 탐색)
                found_rects = []
                threshold = 0.7 # 인식 감도
                
                for scale in np.linspace(0.5, 1.5, 11):
                    resized_t = cv2.resize(template, None, fx=scale, fy=scale)
                    if resized_t.shape[0] > img_gray.shape[0] or resized_t.shape[1] > img_gray.shape[1]:
                        continue
                        
                    res = cv2.matchTemplate(img_gray, resized_t, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= threshold)
                    
                    t_w, t_h = resized_t.shape[::-1]
                    for pt in zip(*loc[::-1]):
                        # PDF 좌표계로 변환
                        pdf_x, pdf_y = pt[0] / zoom, pt[1] / zoom
                        pdf_w, pdf_h = t_w / zoom, t_h / zoom
                        
                        # 중복 좌표 제거
                        if not any(abs(pdf_x - ex) < 15 and abs(pdf_y - ey) < 15 for ex, ey, _, _ in found_rects):
                            found_rects.append((pdf_x, pdf_y, pdf_w, pdf_h))

                # [4단계] 찾은 위치에 V 기입 (검정색 및 우측 이동 반영)
                for (x, y, w, h) in found_rects:
                    # x + 20으로 우측 이동, color=(0,0,0) 검정색
                    page.insert_text((x + 20, y + h - 3), "V", 
                                     fontname="kor", fontsize=21, color=(0, 0, 0))

                print(f"[{page_index+1}p] 시각 분석 결과 {len(found_rects)}개 구역 발견 및 체크 완료")

                # [5단계] 2페이지 성명 기입 (우측 이동 반영)
                if page_index == 1:
                    p2_words = page.get_text("words")
                    for w in p2_words:
                        if target_name in w[4]:
                            # w[0] + 105로 서명란 위치 조정
                            page.insert_text((w[0] + 105, w[1] + 10), target_name, 
                                             fontname="kor", fontsize=11, color=(0, 0, 0))
                            break

            # [6단계] 고효율 압축 저장 (오류 원인 linear 제거)
            save_name = f"{os.path.splitext(file_name)[0]}_날짜보정완료_{time.strftime('%H%M%S')}.pdf"
            doc.save(
                os.path.join(current_dir, save_name),
                garbage=4,     # 미사용 및 중복 객체 제거
                deflate=True,  # 텍스트/이미지 스트림 압축
                clean=True     # 문서 구조 정리 최적화
            )
            print(f"저장 성공 (용량 최적화 완료): {save_name}")

        except Exception:
            traceback.print_exc()
        finally:
            if doc: doc.close()

if __name__ == "__main__":
    try:
        process_pdf_multi_scale()
    finally:
        print("\n" + "="*30)
        input("작업 완료. 엔터를 누르면 종료됩니다...")