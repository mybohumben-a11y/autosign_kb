import os
import io
import fitz  # PyMuPDF
import cv2
import numpy as np
import streamlit as st
import datetime

def process_kb_pdf(uploaded_file, template_path, font_path="NanumGothic.ttf"):
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

def process_meritz_pdf(uploaded_file, font_path="NanumGothic.ttf"):
    # BytesIO를 통해 메모리에서 PDF 로드
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 오늘 날짜 정보 (2자리)
    now = datetime.datetime.now()
    year_str = now.strftime("%y")
    month_str = now.strftime("%m")
    day_str = now.strftime("%d")

    # 기준 이름 좌표 저장을 위한 변수
    ref_name_coords = None
    target_name = ""

    # 1. 2페이지에서 "동의자" 찾기 및 이름 기준 좌표 획득
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

    for page in doc:
        # 한글 폰트 삽입
        page.insert_font(fontname="kor", fontfile=font_path)

        # 2. 모든 페이지: "동의함" 찾아 V 표시 (크기 21)
        found_agrees = page.search_for("동의함")
        for rect in found_agrees:
            # 위치: x0 + 8, y1 + 18
            v_x = rect.x0 + 8
            v_y = rect.y1 + 18
            page.insert_text((v_x, v_y), "V", fontname="kor", fontsize=21, color=(0, 0, 0))

    # 3. 2페이지: 이름 및 날짜 쓰기 (크기 11)
    if len(doc) >= 2:
        page2 = doc[1]
        
        # 이름 쓰기 (기억해둔 좌표 기준)
        if ref_name_coords and target_name:
            nx, ny = ref_name_coords
            page2.insert_text((nx + 85, ny + 10), target_name, fontname="kor", fontsize=11, color=(0, 0, 0))

        # 날짜 쓰기 ("동의일자" 찾기)
        date_labels = page2.search_for("동의일자")
        if date_labels:
            # 첫 번째 검색된 영역 기준
            rect = date_labels[0]
            x1, y1 = rect.x1, rect.y1
            
            # 년, 월, 일 기입
            page2.insert_text((x1 + 80, y1 - 2), year_str, fontname="kor", fontsize=11, color=(0, 0, 0))
            page2.insert_text((x1 + 115, y1 - 2), month_str, fontname="kor", fontsize=11, color=(0, 0, 0))
            page2.insert_text((x1 + 155, y1 - 2), day_str, fontname="kor", fontsize=11, color=(0, 0, 0))

    # 고효율 압축 저장
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

    # 대상 보험사 선택
    insurance_company = st.radio(
        "보험사를 선택하세요:",
        ("KB손해보험", "메리츠화재"),
        horizontal=True
    )
    
    # 템플릿 이미지 및 폰트 파일 경로
    template_path = "image_3664f7.png"
    font_path = "NanumGothic.ttf"
    
    if insurance_company == "KB손해보험" and not os.path.exists(template_path):
        st.error(f"서버에 필수 파일이 없습니다: `{template_path}`")
        st.stop()
    if not os.path.exists(font_path):
        st.error(f"서버에 필수 폰트 파일이 없습니다: `{font_path}`")
        st.stop()
        
    uploaded_file = st.file_uploader(f"[{insurance_company}] 동의서 PDF를 여기에 드래그하세요", type=["pdf"])
    
    if uploaded_file is not None:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext != ".pdf":
            st.error("앗! PDF 파일만 업로드할 수 있습니다. 다시 확인해 주세요.")
        else:
            with st.spinner(f"AI가 {insurance_company} 동의 항목을 분석하고 있습니다..."):
                try:
                    if insurance_company == "KB손해보험":
                        processed_pdf = process_kb_pdf(uploaded_file, template_path, font_path)
                        company_suffix = "KB손보"
                    else:
                        processed_pdf = process_meritz_pdf(uploaded_file, font_path)
                        company_suffix = "메리츠화재"
                    
                    st.success("✅ 처리 완료!")
                    
                    # 다운로드 버튼 (보험사 이름 추가)
                    original_name = os.path.splitext(uploaded_file.name)[0]
                    download_name = f"{original_name}_{company_suffix}_완성.pdf"
                    
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

