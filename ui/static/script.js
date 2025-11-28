document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const indexForm = document.getElementById('index-form');
    const queryForm = document.getElementById('query-form');

    const uploadCard = document.getElementById('upload-card');
    const indexCard = document.getElementById('index-card');
    const queryCard = document.getElementById('query-card');
    const resultsCard = document.getElementById('results-card');

    const fileInput = document.getElementById('file-input');
    const uploadTrigger = document.getElementById('upload-trigger');
    const uploadSelected = document.getElementById('upload-selected');
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
    const analyticsHint = document.getElementById('analytics-hint');
    const analyticsEmpty = document.getElementById('analytics-empty');
    const analyticsGrid = document.getElementById('analytics-grid');
    const analyticsUpdated = document.getElementById('analytics-updated');
    const summaryIndexes = document.getElementById('summary-indexes');
    const summaryQueries = document.getElementById('summary-queries');
    const lightbox = document.getElementById('chart-lightbox');
    const lightboxImage = document.getElementById('lightbox-image');
    const lightboxCaption = document.getElementById('lightbox-caption');

    const chartKeys = ['latency', 'recall', 'trace'];
    const chartBodies = {};
    const chartImages = {};
    const chartFallbacks = {};
    const chartCaptions = {};
    const analyticsHintDefault = analyticsHint ? analyticsHint.textContent : '';

    chartKeys.forEach((key) => {
        const figure = document.querySelector(`[data-chart-key="${key}"]`);
        chartCaptions[key] = figure ? figure.querySelector('figcaption') : null;
        const body = figure ? figure.querySelector('.chart-figure-body') : null;
        chartBodies[key] = body;
        chartImages[key] = document.getElementById(`chart-img-${key}`);
        chartFallbacks[key] = body ? body.querySelector('.chart-fallback') : null;
    });

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

    if (uploadTrigger && fileInput) {
        uploadTrigger.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', () => {
            const files = fileInput.files;
            if (!files || !files.length) {
                if (uploadSelected) {
                    uploadSelected.textContent = '尚未选择文件';
                }
                return;
            }
            const name = files[0].name;
            if (uploadSelected) {
                uploadSelected.textContent = name;
            }
            if (uploadForm) {
                uploadForm.dispatchEvent(new Event('submit', { cancelable: true }));
            }
        });
    }

    uploadForm.addEventListener('submit', (event) => {
        event.preventDefault();
        const files = fileInput.files;
        if (!files || !files.length) {
            setStatus(uploadStatus, '请选择一个 CSV 文件。', true);
            return;
        }
        const selectedFile = files[0];

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
    formData.append('file', selectedFile);

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
                setStatus(uploadStatus, `已上传：${selectedFile.name}`);
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

        xhr.onloadend = () => {
            fileInput.value = '';
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

    const updateChartFigure = (key, meta) => {
        const body = chartBodies[key];
        const img = chartImages[key];
        const fallback = chartFallbacks[key];
        if (!body || !img) {
            return;
        }
        const defaultMsg = fallback ? (fallback.dataset.defaultMessage || fallback.textContent || '') : '';
        if (!meta || !meta.url) {
            img.removeAttribute('src');
            body.dataset.state = 'empty';
            if (fallback) {
                fallback.textContent = meta && meta.message ? meta.message : defaultMsg;
            }
            return;
        }
        const cacheTag = meta.version || Date.now();
        img.src = `${meta.url}?v=${cacheTag}`;
        if (meta.alt) {
            img.alt = meta.alt;
        }
        body.dataset.state = meta.has_data ? 'ready' : 'empty';
        if (fallback) {
            fallback.textContent = meta.has_data ? defaultMsg : (meta.message || defaultMsg);
        }
    };

    const openLightbox = (key) => {
        if (!lightbox || !lightboxImage || !lightboxCaption) {
            return;
        }
        const img = chartImages[key];
        const body = chartBodies[key];
        if (!img || !body || body.dataset.state !== 'ready' || !img.src) {
            return;
        }
        const caption = chartCaptions[key] ? chartCaptions[key].textContent : '';
        lightboxImage.src = img.src;
        lightboxCaption.textContent = caption || 'Analytics Preview';
        lightbox.setAttribute('aria-hidden', 'false');
        lightbox.classList.add('visible');
        document.body.classList.add('no-scroll');
    };

    const closeLightbox = () => {
        if (!lightbox || !lightbox.classList.contains('visible')) {
            return;
        }
        lightbox.classList.remove('visible');
        lightbox.setAttribute('aria-hidden', 'true');
        lightboxImage.removeAttribute('src');
        document.body.classList.remove('no-scroll');
    };

    const bindAnalyticsFigureEvents = () => {
        chartKeys.forEach((key) => {
            const figure = document.querySelector(`[data-chart-key="${key}"]`);
            if (!figure) {
                return;
            }
            const handleActivate = () => openLightbox(key);
            figure.addEventListener('click', handleActivate);
            figure.addEventListener('keydown', (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    handleActivate();
                }
            });
        });
    };

    const initLightbox = () => {
        if (!lightbox) {
            return;
        }
        lightbox.addEventListener('click', (event) => {
            const target = event.target;
            if ((target.dataset && target.dataset.close === 'true') || target === lightbox) {
                closeLightbox();
            }
        });
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeLightbox();
            }
        });
    };

    bindAnalyticsFigureEvents();
    initLightbox();

    const loadAnalytics = async ({ silent = false } = {}) => {
        if (!silent) {
            setStatus(analyticsStatus, '分析数据加载中...');
        }
        try {
            const payload = await fetchJson(`/analytics?t=${Date.now()}`);
            const images = (payload && payload.images) || {};
            const summary = (payload && payload.summary) || {};

            if (analyticsUpdated) {
                const timestamp = summary.generated_at ? new Date(summary.generated_at * 1000) : null;
                analyticsUpdated.textContent = timestamp ? timestamp.toLocaleString() : '尚未生成';
            }
            if (summaryIndexes) {
                summaryIndexes.textContent = formatNumber(summary.index_builds || 0);
            }
            if (summaryQueries) {
                summaryQueries.textContent = formatNumber(summary.query_runs || 0);
            }

            let readyCount = 0;
            chartKeys.forEach((key) => {
                const meta = images[key];
                updateChartFigure(key, meta);
                if (meta && meta.has_data) {
                    readyCount += 1;
                }
            });

            if (readyCount === 0) {
                analyticsEmpty.style.display = 'block';
                analyticsEmpty.textContent = '暂无分析图表，请运行查询或构建索引后刷新。';
                if (analyticsHint) {
                    analyticsHint.textContent = analyticsHintDefault || '尚未生成图表';
                }
            } else {
                analyticsEmpty.style.display = 'none';
                if (analyticsHint) {
                    analyticsHint.textContent = '点击任意图卡查看大图';
                }
            }

            if (!silent) {
                setStatus(analyticsStatus, '分析数据已更新。');
            } else {
                setStatus(analyticsStatus, '');
            }
        } catch (error) {
            chartKeys.forEach((key) => updateChartFigure(key, null));
            analyticsEmpty.style.display = 'block';
            analyticsEmpty.textContent = error.message || '无法获取分析数据。';
            if (analyticsHint) {
                analyticsHint.textContent = analyticsHintDefault || '刷新后重试';
            }
            setStatus(analyticsStatus, '分析刷新失败。', true);
        }
    };

    analyticsBtn.addEventListener('click', () => loadAnalytics({ silent: false }));
    loadAnalytics({ silent: true });
});
