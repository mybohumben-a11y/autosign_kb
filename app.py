import os
import io
import fitz  # PyMuPDF
import cv2
import numpy as np
import streamlit as st

def process_pdf(uploaded_file, template_path, font_path="NanumGothic.ttf"):
    # 템플릿 이미지 읽기 (그레이스케일)
    template = cv2.imread(template_path, 0)
    if template is None:
        raise FileNotFoundError("템플릿 이미지(image_3664f7.png)를 찾을 수 없거나 읽을 수 없습니다.")

    # BytesIO를 통해 메모리에서 PDF 로드
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # [1단계] 고객명 추출 (1페이지)
    page1 = doc[0]
    words_p1 = page1.get_text("words")
    # "고객명" 텍스트 뒤에 오는 값을 이름으로 추출
    target_name = None
    for i, w in enumerate(words_p1):
        if "고객명" in w[4]:
            if i + 2 < len(words_p1):
                target_name = words_p1[i+2][4]
            elif i + 1 < len(words_p1):
                target_name = words_p1[i+1][4]
            break
            
    if not target_name:
        target_name = "고객"
        st.warning("'고객명'을 찾지 못하여 기본값 '고객'으로 설정했습니다.")

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

        # [5단계] 2페이지 성명 기입 (우측 이동 반영)
        if page_index == 1:
            p2_words = page.get_text("words")
            for w in p2_words:
                if target_name in w[4]:
                    # w[0] + 105로 서명란 위치 조정
                    page.insert_text((w[0] + 105, w[1] + 10), target_name, 
                                     fontname="kor", fontsize=11, color=(0, 0, 0))
                    break

    # [6단계] 고효율 압축 저장
    output_buffer = io.BytesIO()
    doc.save(
        output_buffer,
        garbage=4,     # 미사용 및 중복 객체 제거
        deflate=True,  # 텍스트/이미지 스트림 압축
        clean=True     # 문서 구조 정리 최적화
    )
    doc.close()
    
    output_buffer.seek(0)
    return output_buffer

def main():
    st.set_page_config(page_title="보험 동의서 자동 완성", page_icon="📝", layout="centered")
    
    st.title("보험 동의서 자동 완성 웹 앱")
    st.markdown("---")
    
    # 템플릿 이미지 및 폰트 파일 경로
    template_path = "image_3664f7.png"
    font_path = "NanumGothic.ttf"
    
    if not os.path.exists(template_path):
        st.error(f"서버에 필수 파일이 없습니다: `{template_path}`")
        st.stop()
    if not os.path.exists(font_path):
        st.error(f"서버에 필수 폰트 파일이 없습니다: `{font_path}`")
        st.stop()
        
    uploaded_file = st.file_uploader("KB손보 동의서 PDF를 여기에 드래그하세요", type=["pdf"])
    
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext != ".pdf":
            st.error("앗! PDF 파일만 업로드할 수 있습니다. 다시 확인해 주세요.")
        else:
            with st.spinner("AI가 동의 항목을 분석하고 있습니다..."):
                try:
                    processed_pdf = process_pdf(uploaded_file, template_path)
                    
                    st.success("✅ 처리 완료!")
                    
                    # 다운로드 버튼
                    original_name = os.path.splitext(uploaded_file.name)[0]
                    download_name = f"{original_name}_완성.pdf"
                    
                    st.download_button(
                        label="결과물 다운로드",
                        data=processed_pdf,
                        file_name=download_name,
                        mime="application/pdf",
                        type="primary"
                    )
                    
                except Exception as e:
                    st.error(f"처리 중 오류가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    main()
