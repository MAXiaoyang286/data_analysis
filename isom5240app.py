from transformers import pipeline
from PIL import Image
import streamlit as st

# Streamlit UI
st.title("111Age Classification using ViT")

# 定义图片大小调整函数
def resize_image_if_needed(image, max_width=1000, max_height=1000):
    """
    如果图片尺寸过大，则调整大小
    
    参数:
        image: PIL Image对象
        max_width: 最大宽度（像素）
        max_height: 最大高度（像素）
        
    返回:
        调整大小后的PIL Image对象
    """
    original_width, original_height = image.size
    
    # 检查图片是否需要调整大小
    if original_width <= max_width and original_height <= max_height:
        st.write(f"✓ 图片尺寸合适: {original_width} x {original_height} 像素")
        return image
    
    # 计算缩放比例
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    scale_ratio = min(width_ratio, height_ratio)
    
    # 计算新尺寸
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    
    st.write(f"⚠️ 图片尺寸过大，正在调整大小...")
    st.write(f"  原始尺寸: {original_width} x {original_height} 像素")
    st.write(f"  目标尺寸: {new_width} x {new_height} 像素")
    
    # 调整图片大小
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

# 加载年龄分类pipeline
try:
    age_classifier = pipeline("image-classification",
                              model="nateraw/vit-age-classifier",
                              device=-1)  # 强制使用CPU以避免GPU内存问题
    st.write("✓ 年龄分类模型加载成功")
except Exception as e:
    st.error(f"❌ 模型加载失败: {str(e)}")
    st.stop()

# 指定图片路径
image_path = "SAM_2253.jpg"

try:
    # 打开并处理图片
    st.write(f"正在打开图片: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # 显示原始图片
    st.write("原始图片预览:")
    st.image(image, caption=f"原始图片 ({image.size[0]} x {image.size[1]} 像素)", 
             use_column_width=True)
    
    # 调整图片大小（如果需要）
    image = resize_image_if_needed(image, max_width=800, max_height=800)
    
    # 显示调整后的图片
    st.write("调整后图片预览:")
    st.image(image, caption=f"处理后的图片 ({image.size[0]} x {image.size[1]} 像素)", 
             use_column_width=True)
    
    # 分类年龄
    with st.spinner("正在进行年龄分类分析..."):
        age_predictions = age_classifier(image)
        
        # 将原始预测结果转换为可排序的格式
        predictions_list = []
        for pred in age_predictions:
            if isinstance(pred, dict):
                predictions_list.append(pred)
        
        # 按置信度排序
        if predictions_list:
            age_predictions = sorted(predictions_list, key=lambda x: x['score'], reverse=True)
        else:
            st.error("❌ 未能获取有效的年龄预测结果")
            st.stop()
        
        # 显示结果
        st.write("## 年龄分类结果")
        st.write("**预测年龄范围:**")
        
        # 显示最高置信度的结果
        top_prediction = age_predictions[0]
        
        # 使用更美观的显示方式
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="年龄范围",
                value=top_prediction['label']
            )
        with col2:
            st.metric(
                label="置信度",
                value=f"{top_prediction['score']:.2%}"
            )
        
        # 显示所有预测结果（可选）
        with st.expander("查看所有预测结果"):
            st.write("**所有可能的年龄范围预测:**")
            for i, pred in enumerate(age_predictions, 1):
                st.write(f"{i}. {pred['label']}: {pred['score']:.2%}")
        
        st.success("✓ 分析完成！")
        
except FileNotFoundError:
    st.error(f"❌ 找不到图片文件: {image_path}")
    st.write("请确保图片文件存在且路径正确。")
except Exception as e:
    st.error(f"❌ 处理图片时发生错误: {str(e)}")
    st.write("可能的原因:")
    st.write("1. 图片文件损坏")
    st.write("2. 图片格式不受支持")
    st.write("3. 内存不足（请尝试使用更小的图片）")
