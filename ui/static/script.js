document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const indexForm = document.getElementById('index-form');
    const queryForm = document.getElementById('query-form');

    const uploadCard = document.getElementById('upload-card');
    const indexCard = document.getElementById('index-card');
    const queryCard = document.getElementById('query-card');
    const resultsCard = document.getElementById('results-card');

    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');
    const uploadProgress = document.getElementById('upload-progress');
    const uploadProgressContainer = document.getElementById('upload-progress-container');
    const uploadMeta = document.getElementById('upload-meta');

    const sparsitySlider = document.getElementById('sparsity-slider');
    const sparsityValue = document.getElementById('sparsity-value');
    const fieldSelect = document.getElementById('field-select');
    const indexLimitInput = document.getElementById('index-limit');
    const indexProgress = document.getElementById('index-progress');
    const indexProgressContainer = document.getElementById('index-progress-container');
    const indexInfo = document.getElementById('index-info');

    const queryInput = document.getElementById('query-input');
    const queryLimitInput = document.getElementById('query-limit');
    const queryProgress = document.getElementById('query-progress');
    const queryProgressContainer = document.getElementById('query-progress-container');
    const queryWithIndexBtn = document.getElementById('query-with-index');
    const queryWithoutIndexBtn = document.getElementById('query-without-index');

    const resultsInfo = document.getElementById('results-info');
    const resultsTableHead = document.querySelector('#results-table thead');
    const resultsTableBody = document.querySelector('#results-table tbody');

    const analyticsBtn = document.getElementById('analytics-btn');
    const analyticsStatus = document.getElementById('analytics-status');
    const analyticsEmpty = document.getElementById('analytics-empty');
    const analyticsGrid = document.getElementById('analytics-grid');

    let uploadedFilepath = '';
    let datasetMeta = null;
    let indexBuilt = false;

    const setStatus = (node, message, isError = false) => {
        if (!node) {
            return;
        }
        node.textContent = message;
        node.style.color = isError ? '#dc2626' : '#2563eb';
    };

    const setStepEnabled = (card, enabled) => {
        if (!card) {
            return;
        }
        card.classList.toggle('disabled', !enabled);
    };

    const formatNumber = (value) => {
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            return '0';
        }
        return Number(value).toLocaleString();
    };

    const populateFieldOptions = (columns = [], defaultField = '') => {
        if (!fieldSelect) {
            return;
        }
        fieldSelect.innerHTML = '';
        const colList = Array.isArray(columns) ? columns : [];
        if (!colList.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'No columns detected';
            fieldSelect.appendChild(option);
            return;
        }
        colList.forEach((column) => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            if (column === defaultField) {
                option.selected = true;
            }
            fieldSelect.appendChild(option);
        });
        if (defaultField && !colList.includes(defaultField)) {
            fieldSelect.value = defaultField;
        }
    };

    const resetIndexState = () => {
        setStepEnabled(indexCard, !!datasetMeta);
        indexInfo.style.display = 'none';
        indexInfo.innerHTML = '';
        indexBuilt = false;
        queryWithIndexBtn.disabled = true;
    };

    const fetchJson = async (url, options = {}) => {
        const response = await fetch(url, options);
        const text = await response.text();
        let data;
        try {
            data = text ? JSON.parse(text) : {};
        } catch (error) {
            throw new Error('Server returned invalid JSON.');
        }
        if (!response.ok || (data && data.error)) {
            const message = data && data.error ? data.error : response.statusText;
            throw new Error(message || 'Request failed');
        }
        return data;
    };

    const clearChart = (containerId) => {
        const node = document.getElementById(containerId);
        if (node) {
            node.innerHTML = '';
        }
    };

    const setChartMessage = (containerId, message) => {
        const node = document.getElementById(containerId);
        if (node) {
            node.innerHTML = `<div class="chart-empty">${message}</div>`;
        }
    };

    const getD3 = () => {
        const d3Global = window.d3;
        if (!d3Global) {
            console.error('D3.js 未加载，请确认网络连接或脚本路径。');
        }
        return d3Global;
    };

    setStepEnabled(indexCard, false);
    setStepEnabled(queryCard, false);
    queryWithIndexBtn.disabled = true;

    sparsitySlider.addEventListener('input', () => {
        const value = Number.parseFloat(sparsitySlider.value || '0').toFixed(2);
        sparsityValue.textContent = value;
    });

    uploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        if (!fileInput.files || !fileInput.files.length) {
            setStatus(uploadStatus, '请选择一个 CSV 文件。', true);
            return;
        }

        setStepEnabled(indexCard, false);
        setStepEnabled(queryCard, false);
        queryWithIndexBtn.disabled = true;
        indexBuilt = false;
        datasetMeta = null;
        uploadMeta.style.display = 'none';
        uploadMeta.innerHTML = '';
        setStatus(uploadStatus, '上传中...');
        uploadProgressContainer.style.display = 'block';
        uploadProgress.style.width = '0%';

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', (progressEvent) => {
            if (progressEvent.lengthComputable) {
                const percent = Math.min(100, (progressEvent.loaded / progressEvent.total) * 100);
                uploadProgress.style.width = `${percent}%`;
            }
        });

        xhr.onerror = () => {
            setStatus(uploadStatus, '上传失败，请重试。', true);
        };

        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                let payload;
                try {
                    payload = JSON.parse(xhr.responseText || '{}');
                } catch (error) {
                    setStatus(uploadStatus, '服务器返回异常响应。', true);
                    return;
                }
                if (payload.error) {
                    setStatus(uploadStatus, payload.error, true);
                    return;
                }
                uploadedFilepath = payload.filepath;
                datasetMeta = payload.metadata || null;
                setStatus(uploadStatus, `已上传：${fileInput.files[0].name}`);
                if (datasetMeta) {
                    const columns = datasetMeta.columns || [];
                    uploadMeta.innerHTML = `
                        <div><strong>行数：</strong>${formatNumber(datasetMeta.rows)}</div>
                        <div><strong>列数：</strong>${columns.length}</div>
                        <div><strong>推断文本列：</strong>${datasetMeta.text_field || '未检测'}</div>
                        <div><strong>列名：</strong>${columns.join(', ')}</div>
                    `;
                    uploadMeta.style.display = 'block';
                    populateFieldOptions(columns, datasetMeta.text_field);
                }
                setStepEnabled(indexCard, true);
                setStepEnabled(queryCard, true);
                queryWithIndexBtn.disabled = true;
            } else {
                setStatus(uploadStatus, xhr.statusText || '上传失败。', true);
            }
        };

        xhr.open('POST', '/upload', true);
        xhr.send(formData);
    });

    indexForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        if (!uploadedFilepath) {
            alert('请先上传数据文件。');
            return;
        }

        indexInfo.style.display = 'none';
        indexInfo.innerHTML = '';
        indexProgressContainer.style.display = 'block';
        indexProgress.style.width = '0%';
        setStatus(analyticsStatus, '', false);

        const controls = indexForm.querySelectorAll('input, select, button');
        controls.forEach((node) => {
            node.disabled = true;
        });

        let progress = 0;
        const interval = window.setInterval(() => {
            progress = Math.min(100, progress + 6);
            indexProgress.style.width = `${progress}%`;
            if (progress >= 100) {
                window.clearInterval(interval);
            }
        }, 160);

        try {
            const payload = await fetchJson('/build_index', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filepath: uploadedFilepath,
                    sparsity: sparsitySlider.value,
                    field: fieldSelect.value,
                    limit: indexLimitInput.value
                })
            });

            indexProgress.style.width = '100%';
            window.clearInterval(interval);
            indexProgressContainer.style.display = 'none';

            const details = payload.details || {};
            const sizeKb = details.index_size_bytes ? (details.index_size_bytes / 1024).toFixed(1) : payload.index_size;
            indexInfo.innerHTML = `
                <div><strong>索引字段：</strong>${details.field || fieldSelect.value || '未指定'}</div>
                <div><strong>索引行数：</strong>${formatNumber(details.rows_indexed)}</div>
                <div><strong>稀疏度：</strong>${Number.parseFloat(details.sparsity ?? sparsitySlider.value).toFixed(2)}</div>
                <div><strong>索引大小：</strong>${sizeKb} KB</div>
            `;
            indexInfo.style.display = 'block';
            indexBuilt = true;
            queryWithIndexBtn.disabled = false;
            setStatus(analyticsStatus, '索引已构建，可刷新分析。');
        } catch (error) {
            window.clearInterval(interval);
            indexProgressContainer.style.display = 'none';
            alert(`构建索引失败: ${error.message}`);
        } finally {
            controls.forEach((node) => {
                node.disabled = false;
            });
        }
    });

    const runQuery = async (useIndex) => {
        const queryText = (queryInput.value || '').trim();
        if (!queryText) {
            alert('请输入查询表达式。');
            return;
        }
        if (!uploadedFilepath) {
            alert('请先上传数据文件。');
            return;
        }
        if (useIndex && !indexBuilt) {
            alert('请先构建索引后再使用索引查询。');
            return;
        }

        resultsCard.style.display = 'none';
        resultsTableHead.innerHTML = '';
        resultsTableBody.innerHTML = '';
        resultsInfo.innerHTML = '';

        queryProgressContainer.style.display = 'block';
        queryProgress.style.width = '0%';
        queryWithIndexBtn.disabled = true;
        queryWithoutIndexBtn.disabled = true;

        let progress = 0;
        const interval = window.setInterval(() => {
            progress = Math.min(100, progress + 12);
            queryProgress.style.width = `${progress}%`;
            if (progress >= 100) {
                window.clearInterval(interval);
            }
        }, 180);

        try {
            const payload = await fetchJson('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: queryText,
                    use_index: useIndex,
                    limit: queryLimitInput.value
                })
            });

            queryProgress.style.width = '100%';
            window.clearInterval(interval);
            queryProgressContainer.style.display = 'none';

            displayResults(payload, useIndex);
            loadAnalytics({ silent: true });
        } catch (error) {
            window.clearInterval(interval);
            queryProgressContainer.style.display = 'none';
            alert(`查询失败: ${error.message}`);
        } finally {
            queryWithIndexBtn.disabled = !indexBuilt;
            queryWithoutIndexBtn.disabled = false;
        }
    };

    queryWithIndexBtn.addEventListener('click', () => runQuery(true));
    queryWithoutIndexBtn.addEventListener('click', () => runQuery(false));

    const displayResults = (data, wasIndexed) => {
        if (!data) {
            return;
        }
        resultsCard.style.display = 'block';

        const meta = data.metadata || {};
        const rowsMatched = formatNumber(meta.matched_rows);
        const totalRows = formatNumber(meta.total_rows);
        const returnedRows = formatNumber(meta.returned_rows);
        const indexStatus = wasIndexed && meta.use_index ? '使用索引' : '未使用索引';

        resultsInfo.innerHTML = `
            <div><strong>推理耗时：</strong>${data.inference_time || 'N/A'}</div>
            <div><strong>匹配行数：</strong>${rowsMatched} / ${totalRows}</div>
            <div><strong>返回行数：</strong>${returnedRows}</div>
            <div><strong>执行模式：</strong>${indexStatus}</div>
        `;

        resultsTableHead.innerHTML = '';
        resultsTableBody.innerHTML = '';

        const records = Array.isArray(data.results) ? data.results : [];
        if (!records.length) {
            const emptyRow = document.createElement('tr');
            const emptyCell = document.createElement('td');
            emptyCell.colSpan = 1;
            emptyCell.textContent = '没有匹配的记录。';
            emptyRow.appendChild(emptyCell);
            resultsTableBody.appendChild(emptyRow);
        } else {
            const headers = Object.keys(records[0]);
            const headerRow = document.createElement('tr');
            headers.forEach((header) => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            resultsTableHead.appendChild(headerRow);

            records.forEach((row) => {
                const tr = document.createElement('tr');
                headers.forEach((header) => {
                    const td = document.createElement('td');
                    const value = row[header];
                    td.textContent = value === null || value === undefined ? '' : value;
                    tr.appendChild(td);
                });
                resultsTableBody.appendChild(tr);
            });
        }

        renderTrace(data.trace_data || {});
    };

    const renderTrace = (traceData) => {
        const container = document.getElementById('trace-container');
        if (!container) {
            return;
        }
        container.innerHTML = '';
        const d3 = getD3();
        if (!d3) {
            container.innerHTML = '<div class="chart-empty">D3.js 未加载，无法渲染追踪图。</div>';
            return;
        }
        const events = Array.isArray(traceData.traceEvents) ? traceData.traceEvents : [];
        if (!events.length) {
            container.innerHTML = '<div class="chart-empty">暂无追踪数据。</div>';
            return;
        }

        const margin = { top: 18, right: 20, bottom: 40, left: 100 };
        const width = Math.max(320, container.clientWidth - margin.left - margin.right);
        const height = Math.max(120, container.clientHeight - margin.top - margin.bottom);

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        events.sort((a, b) => (a.ts || 0) - (b.ts || 0));
        const tids = Array.from(new Set(events.map((d) => d.tid))).sort((a, b) => a - b);
        const maxTime = d3.max(events, (d) => (d.ts || 0) + (d.dur || 0)) || 0;

        if (!maxTime) {
            container.innerHTML = '<div class="chart-empty">追踪时间轴为空。</div>';
            return;
        }

        const x = d3.scaleLinear()
            .domain([0, maxTime])
            .range([0, width]);

        const y = d3.scaleBand()
            .domain(tids)
            .range([0, height])
            .padding(0.25);

        const color = d3.scaleOrdinal()
            .domain(tids)
            .range(d3.schemeTableau10);

        const xAxis = d3.axisBottom(x)
            .ticks(6)
            .tickFormat((d) => `${(Number(d) / 1000).toFixed(1)} ms`);

        const yAxis = d3.axisLeft(y)
            .tickFormat((d) => `线程 ${d}`);

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);

        svg.append('g')
            .call(yAxis);

        svg.selectAll('rect')
            .data(events)
            .enter()
            .append('rect')
            .attr('x', (d) => x(d.ts || 0))
            .attr('y', (d) => y(d.tid))
            .attr('width', (d) => {
                const start = d.ts || 0;
                const end = start + (d.dur || 0);
                return Math.max(3, x(end) - x(start));
            })
            .attr('height', y.bandwidth())
            .attr('rx', 4)
            .attr('fill', (d) => color(d.tid))
            .append('title')
            .text((d) => `${d.name} · ${(d.dur / 1000).toFixed(2)} ms`);

        svg.selectAll('text.label')
            .data(events)
            .enter()
            .append('text')
            .attr('class', 'label')
            .attr('x', (d) => x(d.ts || 0) + 6)
            .attr('y', (d) => y(d.tid) + y.bandwidth() / 2 + 4)
            .attr('fill', '#ffffff')
            .attr('font-size', '11px')
            .text((d) => d.name);
    };

    const renderLatencyBar = (latencyBar = {}) => {
        clearChart('latency-chart');
        const container = document.getElementById('latency-chart');
        if (!container) {
            return;
        }
        const d3 = getD3();
        if (!d3) {
            setChartMessage('latency-chart', 'D3.js 未加载，无法渲染延迟对比图。');
            return;
        }
        const entries = [
            { label: '使用索引', value: Number(latencyBar.with_index || 0) },
            { label: '未使用索引', value: Number(latencyBar.without_index || 0) }
        ];
        if (!entries.some((item) => item.value > 0)) {
            setChartMessage('latency-chart', '运行一次索引查询和一次基准查询以比较延迟。');
            return;
        }

        const margin = { top: 18, right: 20, bottom: 40, left: 80 };
        const width = Math.max(260, container.clientWidth - margin.left - margin.right);
        const height = Math.max(160, container.clientHeight - margin.top - margin.bottom);

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleBand()
            .domain(entries.map((d) => d.label))
            .range([0, width])
            .padding(0.45);

        const maxLatency = d3.max(entries, (d) => d.value) || 0;
        const y = d3.scaleLinear()
            .domain([0, maxLatency * 1.1])
            .range([height, 0]);

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x));

        svg.append('g')
            .call(d3.axisLeft(y).ticks(6).tickFormat((d) => `${Math.round(d)} ms`));

        const gradient = svg.append('defs')
            .append('linearGradient')
            .attr('id', 'latencyGradient')
            .attr('x1', '0%')
            .attr('x2', '0%')
            .attr('y1', '0%')
            .attr('y2', '100%');
        gradient.append('stop').attr('offset', '0%').attr('stop-color', '#38bdf8');
        gradient.append('stop').attr('offset', '100%').attr('stop-color', '#1d4ed8');

        svg.selectAll('.bar')
            .data(entries)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', (d) => x(d.label))
            .attr('y', (d) => y(d.value))
            .attr('width', x.bandwidth())
            .attr('height', (d) => height - y(d.value))
            .attr('rx', 8)
            .attr('fill', 'url(#latencyGradient)')
            .append('title')
            .text((d) => `${d.label}: ${d.value.toFixed(1)} ms`);

        svg.selectAll('.value-label')
            .data(entries)
            .enter()
            .append('text')
            .attr('class', 'value-label')
            .attr('x', (d) => x(d.label) + x.bandwidth() / 2)
            .attr('y', (d) => y(d.value) - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', '#1f2937')
            .attr('font-weight', '600')
            .text((d) => `${d.value.toFixed(1)} ms`);
    };

    const renderDiffScatter = (diffPoints = []) => {
        clearChart('diff-chart');
        const container = document.getElementById('diff-chart');
        if (!container) {
            return;
        }
        const d3 = getD3();
        if (!d3) {
            setChartMessage('diff-chart', 'D3.js 未加载，无法渲染结果漂移图。');
            return;
        }
        if (!Array.isArray(diffPoints) || !diffPoints.length) {
            setChartMessage('diff-chart', '暂无可比较的结果漂移，请运行索引和非索引查询。');
            return;
        }

        const margin = { top: 18, right: 20, bottom: 50, left: 60 };
        const width = Math.max(260, container.clientWidth - margin.left - margin.right);
        const height = Math.max(180, container.clientHeight - margin.top - margin.bottom);

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear()
            .domain([0, 1])
            .range([0, width]);

        const sparsities = diffPoints.map((d) => Number(d.sparsity || 0));
        const yMin = Math.min(0, d3.min(sparsities) || 0);
        const yMax = Math.max(1, d3.max(sparsities) || 1);
        const y = d3.scaleLinear()
            .domain([yMin, yMax])
            .nice()
            .range([height, 0]);

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5).tickFormat((d) => `${Math.round(d * 100)}%`));

        svg.append('g')
            .call(d3.axisLeft(y).ticks(6).tickFormat((d) => `${(d * 100).toFixed(0)}%`));

        svg.selectAll('circle')
            .data(diffPoints)
            .enter()
            .append('circle')
            .attr('cx', (d) => x(d.diff_ratio || 0))
            .attr('cy', (d) => y(Number(d.sparsity || 0)))
            .attr('r', 6)
            .attr('fill', '#f97316')
            .attr('fill-opacity', 0.75)
            .attr('stroke', '#ea580c')
            .attr('stroke-width', 1.5)
            .append('title')
            .text((d) => `${(d.diff_ratio * 100).toFixed(1)}% 差异 · 稀疏度 ${(Number(d.sparsity) * 100).toFixed(1)}%\n${d.query}`);

        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('transform', `translate(${width / 2}, ${height + margin.bottom - 10})`)
            .attr('fill', '#475569')
            .text('结果差异比例');

        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('transform', `translate(${-40}, ${height / 2}) rotate(-90)`)
            .attr('fill', '#475569')
            .text('稀疏度');
    };

    const renderSparsityLine = (sparsityCurve = []) => {
        clearChart('sparsity-chart');
        const container = document.getElementById('sparsity-chart');
        if (!container) {
            return;
        }
        const d3 = getD3();
        if (!d3) {
            setChartMessage('sparsity-chart', 'D3.js 未加载，无法渲染稀疏度趋势。');
            return;
        }
        if (!Array.isArray(sparsityCurve) || !sparsityCurve.length) {
            setChartMessage('sparsity-chart', '尚未记录索引构建，构建索引后查看稀疏度趋势。');
            return;
        }

        const data = sparsityCurve
            .map((item) => ({
                timestamp: new Date((item.timestamp || 0) * 1000),
                sparsity: Number(item.sparsity || 0),
                size: Number(item.index_size_kb || 0)
            }))
            .sort((a, b) => a.timestamp - b.timestamp);

        const margin = { top: 18, right: 40, bottom: 40, left: 60 };
        const width = Math.max(260, container.clientWidth - margin.left - margin.right);
        const height = Math.max(180, container.clientHeight - margin.top - margin.bottom);

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleTime()
            .domain(d3.extent(data, (d) => d.timestamp))
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, Math.max(1, d3.max(data, (d) => d.sparsity) || 1)])
            .range([height, 0]);

        const line = d3.line()
            .x((d) => x(d.timestamp))
            .y((d) => y(d.sparsity))
            .curve(d3.curveMonotoneX);

        svg.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', '#10b981')
            .attr('stroke-width', 2.5)
            .attr('d', line);

        svg.selectAll('circle')
            .data(data)
            .enter()
            .append('circle')
            .attr('cx', (d) => x(d.timestamp))
            .attr('cy', (d) => y(d.sparsity))
            .attr('r', 5)
            .attr('fill', '#047857')
            .append('title')
            .text((d) => `${d.timestamp.toLocaleString()}\n稀疏度 ${(d.sparsity * 100).toFixed(1)}%\n索引大小 ${d.size.toFixed(1)} KB`);

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(4));

        svg.append('g')
            .call(d3.axisLeft(y).ticks(6).tickFormat((d) => `${(d * 100).toFixed(0)}%`));

        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('transform', `translate(${width / 2}, ${height + margin.bottom - 10})`)
            .attr('fill', '#475569')
            .text('索引构建时间');

        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('transform', `translate(${-40}, ${height / 2}) rotate(-90)`)
            .attr('fill', '#475569')
            .text('稀疏度');
    };

    const renderKvTimeline = (kvTimeline = []) => {
        clearChart('kv-chart');
        const container = document.getElementById('kv-chart');
        if (!container) {
            return;
        }
        const d3 = getD3();
        if (!d3) {
            setChartMessage('kv-chart', 'D3.js 未加载，无法渲染 KV 时间线。');
            return;
        }
        if (!Array.isArray(kvTimeline) || !kvTimeline.length) {
            setChartMessage('kv-chart', '运行查询以收集 KV Cache 传输与计算时间。');
            return;
        }

        const data = kvTimeline
            .map((item) => ({
                timestamp: new Date((item.timestamp || 0) * 1000),
                transfer: Number(item.transfer_ms || 0),
                compute: Number(item.compute_ms || 0),
                useIndex: Boolean(item.use_index)
            }))
            .sort((a, b) => a.timestamp - b.timestamp);

        const margin = { top: 24, right: 60, bottom: 45, left: 70 };
        const width = Math.max(260, container.clientWidth - margin.left - margin.right);
        const height = Math.max(200, container.clientHeight - margin.top - margin.bottom);

        const svg = d3.select(container)
            .append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleTime()
            .domain(d3.extent(data, (d) => d.timestamp))
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, Math.max(1, d3.max(data, (d) => Math.max(d.transfer, d.compute)) || 1)])
            .nice()
            .range([height, 0]);

        const line = (key, color) => {
            const generator = d3.line()
                .x((d) => x(d.timestamp))
                .y((d) => y(d[key]))
                .curve(d3.curveMonotoneX);
            svg.append('path')
                .datum(data)
                .attr('fill', 'none')
                .attr('stroke', color)
                .attr('stroke-width', 2)
                .attr('d', generator);

            svg.selectAll(`circle.${key}`)
                .data(data)
                .enter()
                .append('circle')
                .attr('class', key)
                .attr('cx', (d) => x(d.timestamp))
                .attr('cy', (d) => y(d[key]))
                .attr('r', 4)
                .attr('fill', color)
                .append('title')
                .text((d) => `${d.timestamp.toLocaleString()}\n${key === 'transfer' ? '传输' : '计算'}: ${d[key].toFixed(2)} ms\n${d.useIndex ? '使用索引' : '未使用索引'}`);
        };

        line('transfer', '#2563eb');
        line('compute', '#facc15');

        svg.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(4));

        svg.append('g')
            .call(d3.axisLeft(y).ticks(6).tickFormat((d) => `${d.toFixed(0)} ms`));

        const legend = svg.append('g')
            .attr('transform', `translate(${width - 120}, 0)`);

        legend.append('rect')
            .attr('x', 0)
            .attr('y', 0)
            .attr('width', 12)
            .attr('height', 12)
            .attr('fill', '#2563eb');
        legend.append('text')
            .attr('x', 20)
            .attr('y', 10)
            .attr('fill', '#475569')
            .text('传输时间');

        legend.append('rect')
            .attr('x', 0)
            .attr('y', 20)
            .attr('width', 12)
            .attr('height', 12)
            .attr('fill', '#facc15');
        legend.append('text')
            .attr('x', 20)
            .attr('y', 30)
            .attr('fill', '#475569')
            .text('计算时间');
    };

    const loadAnalytics = async ({ silent = false } = {}) => {
        if (!silent) {
            setStatus(analyticsStatus, '分析数据加载中...');
        }
        try {
            const data = await fetchJson('/analytics');
            const { latency_bar, diff_points, sparsity_curve, kv_timeline } = data || {};

            renderLatencyBar(latency_bar || {});
            renderDiffScatter(diff_points || []);
            renderSparsityLine(sparsity_curve || []);
            renderKvTimeline(kv_timeline || []);

            const hasLatency = latency_bar && (latency_bar.with_index || latency_bar.without_index);
            const hasDiff = Array.isArray(diff_points) && diff_points.length;
            const hasSparsity = Array.isArray(sparsity_curve) && sparsity_curve.length;
            const hasTimeline = Array.isArray(kv_timeline) && kv_timeline.length;

            if (!hasLatency && !hasDiff && !hasSparsity && !hasTimeline) {
                analyticsEmpty.style.display = 'block';
                analyticsEmpty.textContent = '暂无分析数据。请运行查询后刷新。';
            } else {
                analyticsEmpty.style.display = 'none';
            }

            if (!silent) {
                setStatus(analyticsStatus, '分析数据已更新。');
            }
        } catch (error) {
            setChartMessage('latency-chart', '无法渲染延迟图表。');
            setChartMessage('diff-chart', '无法渲染结果漂移图表。');
            setChartMessage('sparsity-chart', '无法渲染稀疏度趋势图。');
            setChartMessage('kv-chart', '无法渲染 KV 时间线。');
            analyticsEmpty.style.display = 'block';
            analyticsEmpty.textContent = error.message || '无法获取分析数据。';
            setStatus(analyticsStatus, '分析刷新失败。', true);
        }
    };

    analyticsBtn.addEventListener('click', () => loadAnalytics({ silent: false }));
    loadAnalytics({ silent: true });
});
