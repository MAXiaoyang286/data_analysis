import streamlit as st
import time
import tempfile
import os
from pathlib import Path
import numpy as np
from transformers import pipeline
import soundfile as sf
import librosa
import plotly.graph_objects as go
from datetime import datetime

# 页面配置
st.set_page_config(
    page_title="TTS模型比较器 - 电影评价",
    page_icon="🎬",
    layout="wide"
)

# 标题和说明
st.title("🎬 TTS模型比较器 - 电影评价音频生成")
st.markdown("""
比较三个Hugging Face TTS模型生成的电影评价音频质量和运行时间
""")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # 模型选择
    st.subheader("模型选择")
    use_model1 = st.checkbox("Meta MMS", value=True, help="facebook/mms-tts-eng")
    use_model2 = st.checkbox("VITS Lite", value=True, help="lxyang/vits")
    use_model3 = st.checkbox("Jets ONNX", value=True, help="NeuML/ljspeech-jets-onnx")
    
    # 高级设置
    st.subheader("高级设置")
    auto_play = st.checkbox("自动播放音频", value=True)
    show_metrics = st.checkbox("显示技术指标", value=False)
    save_audio = st.checkbox("保存音频文件", value=True)
    
    st.divider()
    st.markdown("""
    ### 📊 评估维度
    1. **运行时间**: 模型加载和推理时间
    2. **音频质量**: 主观听感比较
    3. **技术指标**: SNR, 时长等
    """)

# 定义模型
MODELS = [
    ("facebook/mms-tts-eng", "Meta MMS", "🔵", use_model1),
    ("lxyang/vits", "VITS Lite", "🟢", use_model2),
    ("NeuML/ljspeech-jets-onnx", "Jets ONNX", "🟠", use_model3)
]

# 活跃模型
active_models = [(id, name, icon) for id, name, icon, active in MODELS if active]

# 示例电影评价
sample_reviews = [
    "The film's cinematography was absolutely breathtaking, with each frame meticulously composed like a painting.",
    "While the acting performances were superb across the board, the plot felt somewhat predictable and lacked originality in the third act.",
    "A visual masterpiece that will undoubtedly stand the test of time, though the pacing in the second half dragged considerably.",
    "The emotional depth of the characters, combined with a hauntingly beautiful score, created an unforgettable cinematic experience.",
    "Despite a strong opening, the movie fails to maintain its momentum, resulting in a finale that feels rushed and unsatisfying."
]

# 输入区域
st.subheader("🎤 输入电影评价")
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "输入您的电影评价（英文）:",
        value=sample_reviews[0],
        height=100,
        placeholder="输入英文电影评价，例如: 'The acting was phenomenal, but the plot had some inconsistencies...'"
    )

with col2:
    st.write("**示例评价:**")
    for i, review in enumerate(sample_reviews[:3]):
        if st.button(f"示例 {i+1}", key=f"sample_{i}"):
            st.session_state.text_input = review
            st.rerun()

# 创建输出目录
output_dir = Path("audio_outputs")
output_dir.mkdir(exist_ok=True)

def calculate_audio_metrics(audio_path):
    """计算音频技术指标"""
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        
        # 信噪比（估算）
        signal_power = np.mean(audio**2)
        noise_estimate = np.std(audio - np.mean(audio))
        snr = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10)) if noise_estimate > 0 else 0
        
        # 时长
        duration = len(audio) / sr
        
        # 动态范围
        dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
        
        return {
            "duration": duration,
            "sampling_rate": sr,
            "snr_db": snr,
            "dynamic_range_db": dynamic_range,
            "audio_shape": audio.shape
        }
    except Exception as e:
        return {"error": str(e)}

def generate_tts_audio(model_id, model_name, text, run_id):
    """生成TTS音频"""
    result = {
        "model": model_name,
        "model_id": model_id,
        "success": False
    }
    
    try:
        # 计时：模型加载
        load_start = time.time()
        tts_pipeline = pipeline("text-to-speech", model=model_id)
        load_time = time.time() - load_start
        
        # 计时：推理
        infer_start = time.time()
        audio_output = tts_pipeline(text)
        infer_time = time.time() - infer_start
        
        # 保存音频
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name.replace(' ', '_')}_{run_id}_{timestamp}.wav"
        filepath = output_dir / filename
        
        # 保存为WAV文件
        sf.write(str(filepath), audio_output["audio"], audio_output["sampling_rate"])
        
        # 计算指标
        metrics = calculate_audio_metrics(str(filepath))
        
        result.update({
            "success": True,
            "load_time": load_time,
            "infer_time": infer_time,
            "total_time": load_time + infer_time,
            "filepath": str(filepath),
            "filename": filename,
            "sampling_rate": audio_output["sampling_rate"],
            "audio_duration": len(audio_output["audio"]) / audio_output["sampling_rate"],
            "metrics": metrics
        })
        
    except Exception as e:
        result["error"] = str(e)
    
    return result

# 生成按钮
if st.button("🚀 生成并比较音频", type="primary", use_container_width=True):
    if not user_input.strip():
        st.warning("请输入电影评价文本")
    elif not active_models:
        st.warning("请至少选择一个模型")
    else:
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        run_id = int(time.time())
        
        # 为每个选择的模型生成音频
        for i, (model_id, model_name, icon) in enumerate(active_models):
            status_text.text(f"{icon} 正在生成 {model_name} 的音频...")
            
            result = generate_tts_audio(model_id, model_name, user_input, run_id)
            results.append(result)
            
            progress_bar.progress((i + 1) / len(active_models))
        
        status_text.text("✅ 所有音频生成完成！")
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # 保存结果到session state
        st.session_state.tts_results = results
        st.session_state.current_text = user_input

# 显示比较结果
if "tts_results" in st.session_state and st.session_state.tts_results:
    st.divider()
    st.subheader("📊 比较结果")
    
    results = st.session_state.tts_results
    successful_results = [r for r in results if r["success"]]
    
    if successful_results:
        # 创建标签页
        tab1, tab2, tab3, tab4 = st.tabs(["🎧 音频播放", "📈 性能比较", "🔧 技术指标", "📥 下载"])
        
        with tab1:
            st.write(f"**输入文本:** *{st.session_state.current_text[:100]}...*")
            st.write("---")
            
            cols = st.columns(len(successful_results))
            for idx, (col, result) in enumerate(zip(cols, successful_results)):
                with col:
                    # 模型卡片
                    with st.container():
                        st.markdown(f"### {['🔵', '🟢', '🟠'][idx]} {result['model']}")
                        
                        # 时间指标
                        st.metric("加载时间", f"{result['load_time']:.2f}s")
                        st.metric("推理时间", f"{result['infer_time']:.2f}s")
                        st.metric("总时间", f"{result['total_time']:.2f}s")
                        
                        # 音频信息
                        st.write(f"**音频信息:**")
                        st.write(f"• 时长: {result['audio_duration']:.2f}s")
                        st.write(f"• 采样率: {result['sampling_rate']}Hz")
                        
                        # 音频播放器
                        st.audio(result['filepath'])
                        
                        if show_metrics and "metrics" in result:
                            with st.expander("查看技术指标"):
                                metrics = result["metrics"]
                                if "error" not in metrics:
                                    st.write(f"信噪比: {metrics.get('snr_db', 0):.1f} dB")
                                    st.write(f"动态范围: {metrics.get('dynamic_range_db', 0):.1f} dB")
        
        with tab2:
            # 性能比较图表
            st.subheader("性能比较")
            
            fig = go.Figure(data=[
                go.Bar(name='加载时间', 
                      x=[r['model'] for r in successful_results],
                      y=[r['load_time'] for r in successful_results],
                      marker_color='lightblue'),
                go.Bar(name='推理时间',
                      x=[r['model'] for r in successful_results],
                      y=[r['infer_time'] for r in successful_results],
                      marker_color='lightcoral')
            ])
            
            fig.update_layout(
                title='模型运行时间比较',
                barmode='stack',
                yaxis_title='时间 (秒)',
                xaxis_title='模型',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 性能排名
            st.subheader("性能排名")
            sorted_by_speed = sorted(successful_results, key=lambda x: x['total_time'])
            
            for i, result in enumerate(sorted_by_speed, 1):
                medal = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"{i}."
                st.write(f"{medal} **{result['model']}**: {result['total_time']:.2f}秒 (推理: {result['infer_time']:.2f}s)")
        
        with tab3:
            if any("metrics" in r for r in successful_results):
                st.subheader("音频技术指标")
                
                # 创建指标表格
                metrics_data = []
                for result in successful_results:
                    if "metrics" in result and "error" not in result["metrics"]:
                        metrics_data.append({
                            "Model": result['model'],
                            "Duration (s)": f"{result['audio_duration']:.2f}",
                            "SNR (dB)": f"{result['metrics'].get('snr_db', 0):.1f}",
                            "Dynamic Range (dB)": f"{result['metrics'].get('dynamic_range_db', 0):.1f}",
                            "Sampling Rate (Hz)": result['sampling_rate']
                        })
                
                if metrics_data:
                    st.table(metrics_data)
                else:
                    st.info("无可用技术指标")
            else:
                st.info("启用'显示技术指标'以查看详细指标")
        
        with tab4:
            st.subheader("下载音频文件")
            
            for result in successful_results:
                with open(result['filepath'], 'rb') as f:
                    audio_bytes = f.read()
                
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{result['model']}** - {result['filename']}")
                with col2:
                    st.download_button(
                        label="下载",
                        data=audio_bytes,
                        file_name=result['filename'],
                        mime="audio/wav",
                        key=f"download_{result['model']}"
                    )
                with col3:
                    file_size = os.path.getsize(result['filepath']) / 1024
                    st.write(f"{file_size:.1f} KB")
    
    else:
        st.error("所有模型生成失败，请检查网络连接或模型配置")

# 历史记录
if "tts_results" in st.session_state:
    with st.expander("📋 历史记录"):
        if len(st.session_state.get('history', [])) > 0:
            for i, hist in enumerate(st.session_state.history[-5:][::-1], 1):
                st.write(f"{i}. {hist['text'][:50]}... - {hist['timestamp']}")
        else:
            st.info("暂无历史记录")

# 页脚
st.divider()
st.markdown("""
---
### 📌 模型信息
1. **Meta MMS** (`facebook/mms-tts-eng`): Meta的Massively Multilingual Speech项目英语TTS模型
2. **VITS Lite** (`lxyang/vits`): 基于VITS架构的轻量级模型
3. **Jets ONNX** (`NeuML/ljspeech-jets-onnx`): FastPitch + HiFi-GAN的ONNX优化版本

*注意: 首次运行需要下载模型，请确保网络连接正常*
""")
