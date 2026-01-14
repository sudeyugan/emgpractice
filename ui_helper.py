import streamlit as st
import os
import glob
import data_loader # 需要引用这个来解析文件名

@st.cache_data
def scan_data_folder(root_dir):
    """扫描文件夹，构建 Subject -> Date -> Labels 结构"""
    structure = {}
    file_map = {} 
    
    pattern = os.path.join(root_dir, "*", "*", "RAW_EMG*.csv")
    files = glob.glob(pattern)
    
    for f in files:
        subject, date, label, fname = data_loader.parse_filename_info(f)
        if label is None: continue
        
        if subject not in structure: structure[subject] = {}
        if date not in structure[subject]: structure[subject][date] = set()
        
        structure[subject][date].add(label)
        
        key = (subject, date, label)
        if key not in file_map: file_map[key] = []
        file_map[key].append(f)
        
    return structure, file_map

def render_multiselect_with_all(label, options, key_name, default_first=False):
    """
    带全选功能的多选框组件
    """
    # 1. 初始化 Session State
    if key_name not in st.session_state:
        if default_first and options:
            st.session_state[key_name] = options[:1]
        else:
            st.session_state[key_name] = []
    
    # 2. 数据清洗
    current_selection = st.session_state[key_name]
    valid_selection = [x for x in current_selection if x in options]
    if len(valid_selection) != len(current_selection):
        st.session_state[key_name] = valid_selection
        current_selection = valid_selection
        
    # 3. 状态计算
    is_all_selected = (len(current_selection) == len(options)) and (len(options) > 0)
    
    # 4. 回调函数
    def toggle_all():
        if st.session_state[key_name + '_all']:
            st.session_state[key_name] = options 
        else:
            st.session_state[key_name] = []      

    # 5. 渲染 UI
    c1, c2 = st.columns([0.7, 0.3], gap="small")
    with c1:
        st.markdown(f"**{label}**")
    with c2:
        st.checkbox(
            "全选", 
            value=is_all_selected, 
            key=key_name + '_all', 
            on_change=toggle_all,   
            help=f"勾选以选中所有 {label}"
        )
        
    return st.multiselect(
        label=label, 
        options=options,
        key=key_name,
        label_visibility="collapsed"
    )