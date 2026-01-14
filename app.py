"""
行政处罚决定书类案检索系统
Flask 主应用
"""
import json
import queue
import threading
from io import BytesIO
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response, send_file
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from modules.case_loader import load_config, get_available_types, load_cases_by_type, get_cases_summary
from modules.similarity_search import search_similar_cases, batch_cases_by_chars
from modules.result_merger import merge_and_sort, format_results_for_display

app = Flask(__name__)
# 增加请求大小限制为 50MB，支持大数据量导出
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


@app.route('/')
def index():
    """主页"""
    available_types = get_available_types()
    config = load_config()
    # 如果没有实际数据，显示配置中的类型
    if not available_types:
        available_types = config.get('case_types', [])
    return render_template('index.html', case_types=available_types)


@app.route('/api/types')
def get_types():
    """获取可用的案件类型"""
    available_types = get_available_types()
    config = load_config()
    if not available_types:
        available_types = config.get('case_types', [])
    return jsonify({'types': available_types})


@app.route('/api/type-summary/<case_type>')
def get_type_summary(case_type):
    """获取指定类型的案例统计"""
    summary = get_cases_summary(case_type)
    return jsonify(summary)


@app.route('/api/search-stream', methods=['POST'])
def search_stream():
    """
    执行类案检索（SSE 流式返回进度）
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    manual_case_type = data.get('case_type', '').strip()
    
    if not query:
        return jsonify({'error': '请输入查询内容'}), 400
    
    if not manual_case_type:
        return jsonify({'error': '请选择案件类型'}), 400
    
    config = load_config()
    
    case_type_info = {
        'case_type': manual_case_type,
        'confidence': '高',
        'reason': '用户指定'
    }
    
    case_type = manual_case_type
    
    # 加载对应类型的案例
    cases = load_cases_by_type(case_type)
    
    if not cases:
        return jsonify({
            'case_type_info': case_type_info,
            'message': f'类型 "{case_type}" 下暂无案例数据，请先添加案例文件到 data/{case_type}/ 目录',
            'results': []
        })
    
    # 计算批次数量
    max_chars = config['search']['max_chars_per_request']
    batches = batch_cases_by_chars(cases, max_chars)
    total_batches = len(batches)
    
    def generate():
        # 发送初始信息
        yield f"data: {json.dumps({'type': 'init', 'total_batches': total_batches, 'total_cases': len(cases), 'case_type': case_type})}\n\n"
        
        # 用于线程间通信的队列
        progress_queue = queue.Queue()
        all_results = []
        
        def progress_callback(completed, total, batch_results):
            progress_queue.put({
                'completed': completed,
                'total': total,
                'batch_results': batch_results
            })
        
        # 在后台线程中执行搜索
        def run_search():
            nonlocal all_results
            all_results = search_similar_cases(query, cases, progress_callback)
            progress_queue.put({'done': True})
        
        search_thread = threading.Thread(target=run_search)
        search_thread.start()
        
        # 流式发送进度
        while True:
            try:
                msg = progress_queue.get(timeout=60)
                if msg.get('done'):
                    break
                
                yield f"data: {json.dumps({'type': 'progress', 'completed': msg['completed'], 'total': msg['total'], 'batch_count': len(msg['batch_results'])})}\n\n"
            except queue.Empty:
                break
        
        search_thread.join()
        
        # 归并排序（只保留>=60分的结果）
        min_score = config['search'].get('min_score', 60)
        sorted_results = merge_and_sort(all_results, min_score)
        
        # 格式化结果
        formatted_results = format_results_for_display(sorted_results)
        
        # 发送最终结果
        final_data = {
            'type': 'complete',
            'case_type_info': case_type_info,
            'total_cases': len(cases),
            'results': formatted_results
        }
        yield f"data: {json.dumps(final_data)}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/search', methods=['POST'])
def search():
    """
    执行类案检索（非流式版本，保留兼容性）
    """
    data = request.get_json()
    query = data.get('query', '').strip()
    manual_case_type = data.get('case_type', '').strip()
    
    if not query:
        return jsonify({'error': '请输入查询内容'}), 400
    
    if not manual_case_type:
        return jsonify({'error': '请选择案件类型'}), 400
    
    config = load_config()
    
    case_type_info = {
        'case_type': manual_case_type,
        'confidence': '高',
        'reason': '用户指定'
    }
    
    case_type = manual_case_type
    
    # 加载对应类型的案例
    cases = load_cases_by_type(case_type)
    
    if not cases:
        return jsonify({
            'case_type_info': case_type_info,
            'message': f'类型 "{case_type}" 下暂无案例数据，请先添加案例文件到 data/{case_type}/ 目录',
            'results': []
        })
    
    # 检索相似案例
    search_results = search_similar_cases(query, cases)
    
    # 归并排序（只保留>=60分的结果）
    min_score = config['search'].get('min_score', 60)
    sorted_results = merge_and_sort(search_results, min_score)
    
    # 格式化结果
    formatted_results = format_results_for_display(sorted_results)
    
    return jsonify({
        'case_type_info': case_type_info,
        'total_cases': len(cases),
        'results': formatted_results
    })


@app.route('/api/export-excel', methods=['POST'])
def export_excel():
    """
    导出检索结果为 Excel 文件
    """
    # 支持 JSON 和表单两种提交方式
    if request.is_json:
        data = request.get_json()
    else:
        # 从表单数据解析 JSON
        form_data = request.form.get('data', '{}')
        data = json.loads(form_data)
    
    results = data.get('results', [])
    case_type = data.get('case_type', '类案检索')
    
    if not results:
        return jsonify({'error': '没有可导出的结果'}), 400
    
    # 创建工作簿
    wb = Workbook()
    ws = wb.active
    ws.title = "检索结果"
    
    # 定义样式
    header_font = Font(bold=True, color="FFFFFF", size=12)
    header_fill = PatternFill(start_color="4A90E2", end_color="4A90E2", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    
    cell_alignment = Alignment(vertical="top", wrap_text=True)
    thin_border = Border(
        left=Side(style='thin', color='CCCCCC'),
        right=Side(style='thin', color='CCCCCC'),
        top=Side(style='thin', color='CCCCCC'),
        bottom=Side(style='thin', color='CCCCCC')
    )
    
    # 设置表头
    headers = ['排名', '文件名', '相似度', '案例摘要', '相似理由', '全文内容']
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border
    
    # 设置列宽
    ws.column_dimensions['A'].width = 8   # 排名
    ws.column_dimensions['B'].width = 30  # 文件名
    ws.column_dimensions['C'].width = 10  # 相似度
    ws.column_dimensions['D'].width = 50  # 案例摘要
    ws.column_dimensions['E'].width = 40  # 相似理由
    ws.column_dimensions['F'].width = 80  # 全文内容
    
    # 填充数据
    for row, result in enumerate(results, 2):
        ws.cell(row=row, column=1, value=result.get('rank', row-1)).alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(row=row, column=2, value=result.get('filename', '')).alignment = cell_alignment
        ws.cell(row=row, column=3, value=result.get('similarity_score', '')).alignment = Alignment(horizontal="center", vertical="top")
        ws.cell(row=row, column=4, value=result.get('summary', '')).alignment = cell_alignment
        ws.cell(row=row, column=5, value=result.get('reason', '')).alignment = cell_alignment
        ws.cell(row=row, column=6, value=result.get('content', '')).alignment = cell_alignment
        
        # 添加边框
        for col in range(1, 7):
            ws.cell(row=row, column=col).border = thin_border
    
    # 冻结首行
    ws.freeze_panes = 'A2'
    
    # 保存到内存
    output = BytesIO()
    wb.save(output)
    output.seek(0)
    
    # 生成文件名（使用 URL 编码处理中文）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{case_type}_检索结果_{timestamp}.xlsx'
    
    # 使用 urllib 编码中文文件名
    from urllib.parse import quote
    encoded_filename = quote(filename)
    
    response = Response(
        output.getvalue(),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{encoded_filename}"
    response.headers['X-Filename'] = encoded_filename
    response.headers['Access-Control-Expose-Headers'] = 'X-Filename, Content-Disposition'
    return response


if __name__ == '__main__':
    print("=" * 50)
    print("行政处罚决定书类案检索系统")
    print("=" * 50)
    
    # 检查配置
    try:
        config = load_config()
        if config['openai']['api_key'] == 'your-api-key-here':
            print("\n⚠️  警告: 请先在 config.yaml 中配置您的 OpenAI API Key")
        else:
            print("✓ API Key 已配置")
    except Exception as e:
        print(f"\n❌ 配置文件读取失败: {e}")
    
    # 检查数据
    available_types = get_available_types()
    if available_types:
        print(f"✓ 发现 {len(available_types)} 种案件类型: {', '.join(available_types)}")
    else:
        print("\n⚠️  警告: data 目录下暂无案例数据")
        print("   请将 txt 案例文件放入对应的类型文件夹中，例如:")
        print("   - data/内幕交易/案例1.txt")
        print("   - data/操纵市场/案例2.txt")
    
    print("\n启动服务...")
    print("访问地址: http://localhost:5001")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
