import streamlit as st
import requests
from PIL import Image
import io

# 配置页面
st.set_page_config(
    page_title="Multimodal Search System",
    page_icon="🔍",
    layout="wide"
)

# API配置
API_URL = "http://localhost:8000/api/v1/search"

def main():
    st.title("Multimodal Search System")
    st.write("Search using text, image, or both!")

    # 创建两列布局
    col1, col2 = st.columns(2)

    # 左侧：搜索输入
    with col1:
        st.subheader("Search Input")
        
        # 文本输入
        text_query = st.text_input("Enter text query:", key="text_query")
        
        # 图片上传
        uploaded_file = st.file_uploader("Upload an image (optional):", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # 搜索按钮
        if st.button("Search"):
            if not text_query and uploaded_file is None:
                st.error("Please provide either text or image for search.")
            else:
                with st.spinner("Searching..."):
                    # 准备请求数据
                    files = {}
                    data = {"top_k": 5}
                    
                    if text_query:
                        data["text"] = text_query
                    if uploaded_file:
                        files["image"] = uploaded_file
                    
                    # 发送请求
                    try:
                        response = requests.post(API_URL, data=data, files=files)
                        response.raise_for_status()
                        results = response.json()["results"]
                        
                        # 在右侧显示结果
                        with col2:
                            st.subheader("Search Results")
                            for idx, result in enumerate(results, 1):
                                with st.container():
                                    st.write(f"**Result {idx}** (Score: {result['score']:.2f})")
                                    st.write(result["text"])
                                    if result.get("image_url"):
                                        st.image(result["image_url"], use_column_width=True)
                                    st.divider()
                    
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error during search: {str(e)}")

if __name__ == "__main__":
    main()