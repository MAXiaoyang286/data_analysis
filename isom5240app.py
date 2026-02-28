from transformers import pipeline
from PIL import Image
import streamlit as st

# Streamlit UI
st.title("Age Classification using ViT")

# 加载年龄分类pipeline
# 注意：添加device=-1强制使用CPU，避免GPU内存问题
@st.cache_resource
def load_age_classifier():
    """加载并缓存年龄分类器"""
    return pipeline("image-classification",
                    model="nateraw/vit-age-classifier",
                    device=-1)  # 强制使用CPU

# 图片预处理函数
def preprocess_image(image_path, max_size=(1000, 1000)):
    """
    预处理图片：打开、转换为RGB、调整大小
    Args:
        image_path: 图片文件路径
        max_size: 最大尺寸元组 (宽, 高)，默认(1000, 1000)
    Returns:
        处理后的PIL Image对象
    """
    try:
        # 打开图片并转换为RGB格式
        image = Image.open(image_path).convert("RGB")
        
        # 获取原始尺寸
        original_size = image.size
        st.info(f"Original image size: {original_size[0]} x {original_size[1]} pixels")
        
        # 如果图片尺寸超过最大值，进行缩放
        if original_size[0] > max_size[0] or original_size[1] > max_size[1]:
            st.warning("Image is too large, resizing for better performance...")
            
            # 计算缩放比例
            width_ratio = max_size[0] / original_size[0]
            height_ratio = max_size[1] / original_size[1]
            ratio = min(width_ratio, height_ratio)
            
            # 计算新尺寸
            new_width = int(original_size[0] * ratio)
            new_height = int(original_size[1] * ratio)
            new_size = (new_width, new_height)
            
            # 使用高质量重采样方法调整大小
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.success(f"Image resized to: {new_width} x {new_height} pixels")
        else:
            st.success("Image size is within acceptable range")
        
        return image
        
    except FileNotFoundError:
        st.error(f"Error: File '{image_path}' not found.")
        return None
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# 主程序逻辑
def main():
    # 加载模型
    with st.spinner("Loading age classification model..."):
        age_classifier = load_age_classifier()
    
    # 图片路径
    image_path = "SAM_2253.JPG"
    
    # 预处理图片
    st.subheader("Image Preprocessing")
    processed_image = preprocess_image(image_path, max_size=(1000, 1000))
    
    if processed_image is None:
        st.error("Failed to process image. Please check the file path and format.")
        return
    
    # 显示处理后的图片
    st.subheader("Processed Image Preview")
    st.image(processed_image, caption="Processed Image (Resized if needed)", use_column_width=True)
    
    # 进行分类
    st.subheader("Age Classification Results")
    with st.spinner("Analyzing age..."):
        try:
            # 分类年龄
            age_predictions = age_classifier(processed_image)
            
            # 调试信息（可选）
            if st.checkbox("Show raw predictions (debug)"):
                st.write("Raw predictions:", age_predictions)
            
            # 按置信度排序
            age_predictions = sorted(age_predictions, key=lambda x: x['score'], reverse=True)
            
            # 显示结果
            st.success("Analysis completed!")
            
            # 显示主要预测结果
            top_prediction = age_predictions[0]
            st.write("**Predicted Age Range:**")
            
            # 使用更美观的方式显示
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Age Range", 
                    value=top_prediction['label']
                )
            with col2:
                st.metric(
                    label="Confidence", 
                    value=f"{top_prediction['score']:.2%}"
                )
            
            # 显示所有预测结果
            with st.expander("View all predictions"):
                st.write("**All age predictions (sorted by confidence):**")
                for i, pred in enumerate(age_predictions, 1):
                    st.progress(
                        value=pred['score'],
                        text=f"{i}. {pred['label']}: {pred['score']:.2%}"
                    )
            
            st.write("**done** ✅")
            
        except Exception as e:
            st.error(f"Error during age classification: {str(e)}")
            st.info("Try using a smaller image or check the model compatibility.")

# 运行主程序
if __name__ == "__main__":
    main()
