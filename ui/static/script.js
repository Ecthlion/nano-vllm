
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const indexForm = document.getElementById('index-form');
    const queryForm = document.getElementById('query-form');
    const fileInput = document.getElementById('file-input');
    let uploadedFilepath = '';

    // Sparsity Slider
    const sparsitySlider = document.getElementById('sparsity-slider');
    const sparsityValue = document.getElementById('sparsity-value');
    sparsitySlider.addEventListener('input', () => {
        sparsityValue.textContent = sparsitySlider.value;
    });

    // Preset Buttons
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetId = btn.getAttribute('data-target');
            const value = btn.getAttribute('data-value');
            const target = document.getElementById(targetId);
            if (target) {
                target.value = value;
            }
        });
    });

    // --- Step 1: File Upload ---
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;

        // const progressBar = document.getElementById('upload-progress');
        // const progressContainer = document.getElementById('upload-progress-container');
        const statusP = document.getElementById('upload-status');

        // Check if file exists
        fetch('/check_file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: file.name })
        })
        .then(response => response.json())
        .then(data => {
            if (data.exists) {
                // File exists, simulate upload
                simulateUpload(data.filepath);
            } else {
                // File doesn't exist, proceed with upload
                performUpload(file);
            }
        })
        .catch(error => {
            console.error('Error checking file:', error);
            statusP.textContent = `Error checking file: ${error.message}`;
            statusP.style.color = 'red';
        });

        function simulateUpload(filepath) {
            // progressContainer.style.display = 'block';
            // progressBar.style.width = '0%';
            statusP.textContent = 'Uploading...';

            // let width = 0;
            // const interval = setInterval(() => {
            //     width += 5; // Fast simulation
            //     progressBar.style.width = width + '%';
            //     if (width >= 100) {
            //         clearInterval(interval);
                    statusP.textContent = `File uploaded successfully: ${file.name}`;
                    statusP.style.color = 'green';
                    uploadedFilepath = filepath;
                    document.getElementById('index-section').style.display = 'block';
            //     }
            // }, 20); // Fast interval
        }

        function performUpload(file) {
            const formData = new FormData();
            formData.append('file', file);

            // progressContainer.style.display = 'block';
            // progressBar.style.width = '0%';
            statusP.textContent = 'Uploading...';

            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    // progressBar.style.width = percentComplete + '%';
                }
            });

            xhr.onload = function() {
                if (xhr.status === 200) {
                    const data = JSON.parse(xhr.responseText);
                    // progressBar.style.width = '100%';
                    if (data.error) {
                        statusP.textContent = `Error: ${data.error}`;
                        statusP.style.color = 'red';
                    } else {
                        statusP.textContent = `File uploaded successfully: ${file.name}`;
                        statusP.style.color = 'green';
                        uploadedFilepath = data.filepath;
                        document.getElementById('index-section').style.display = 'block';
                    }
                } else {
                    statusP.textContent = `Error: ${xhr.statusText}`;
                    statusP.style.color = 'red';
                }
            };

            xhr.onerror = function() {
                statusP.textContent = 'Upload failed.';
                statusP.style.color = 'red';
            };

            xhr.open('POST', '/upload', true);
            xhr.send(formData);
        }
    });

    // --- Step 2: Build Index ---
    indexForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const sparsity = sparsitySlider.value;
        const field = document.getElementById('index-field').value;

        // const progressBar = document.getElementById('index-progress');
        // const progressContainer = document.getElementById('index-progress-container');
        const indexInfo = document.getElementById('index-info');

        // progressContainer.style.display = 'block';
        // progressBar.style.width = '0%';
        indexInfo.style.display = 'none';

        // Simulate index building progress
        // let width = 0;
        // const interval = setInterval(() => {
        //     width += 5;
        //     progressBar.style.width = width + '%';
        //     if (width >= 100) {
        //         clearInterval(interval);
        //     }
        // }, 150);

        fetch('/build_index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: uploadedFilepath, sparsity: sparsity, field: field })
        })
        .then(response => response.json())
        .then(data => {
            // clearInterval(interval);
            // progressBar.style.width = '100%';
            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                document.getElementById('index-size').textContent = data.index_size;
                indexInfo.style.display = 'block';
                // document.getElementById('query-card').style.display = 'block'; // Already visible
            }
        })
        .catch(error => {
            // clearInterval(interval);
            alert(`Error: ${error.message}`);
        });
    });

    // --- Step 3: Query ---
    document.getElementById('query-with-index').addEventListener('click', () => runQuery(true));
    document.getElementById('analyse-btn').addEventListener('click', runAnalyse);

    function runAnalyse() {
        const query = document.getElementById('query-input').value;
        if (!query) {
            alert('Please enter a query.');
            return;
        }

        // const progressBar = document.getElementById('query-progress');
        // const progressContainer = document.getElementById('query-progress-container');
        
        // progressContainer.style.display = 'block';
        // progressBar.style.width = '0%';
        // document.getElementById('results-card').style.display = 'none';

        // Simulate progress
        // let width = 0;
        // const interval = setInterval(() => {
        //     width += 5;
        //     progressBar.style.width = width + '%';
        //     if (width >= 100) {
        //         clearInterval(interval);
        //     }
        // }, 500);

        fetch('/analyse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // clearInterval(interval);
            // progressBar.style.width = '100%';
            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                displayAnalyseResults(data);
            }
        })
        .catch(error => {
            // clearInterval(interval);
            alert(`Error: ${error.message}`);
        });
    }

    function runQuery(useIndex) {
        const query = document.getElementById('query-input').value;
        if (!query) {
            alert('Please enter a query.');
            return;
        }

        // const progressBar = document.getElementById('query-progress');
        // const progressContainer = document.getElementById('query-progress-container');
        
        // progressContainer.style.display = 'block';
        // progressBar.style.width = '0%';
        document.getElementById('results-card').style.display = 'none';

        // Simulate query progress
        // let width = 0;
        // const interval = setInterval(() => {
        //     width += 10;
        //     progressBar.style.width = width + '%';
        //     if (width >= 100) {
        //         clearInterval(interval);
        //     }
        // }, 200);

        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, use_index: useIndex })
        })
        .then(response => response.json())
        .then(data => {
            // clearInterval(interval);
            // progressBar.style.width = '100%';
            displayResults(data);
        })
        .catch(error => {
            // clearInterval(interval);
            alert(`Error: ${error.message}`);
        });
    }

    // --- Step 4: Display Results ---
    let currentResults = [];
    let currentPage = 1;
    const rowsPerPage = 2;

    function displayResults(data) {
        document.getElementById('results-card').style.display = 'block';
        document.getElementById('results-info').style.display = 'block';
        document.getElementById('results-table-container').style.display = 'block';
        document.getElementById('profile-chart').style.display = 'block';
        document.getElementById('analyse-chart').style.display = 'none';
        
        // Display inference time
        document.getElementById('inference-time').textContent = data.inference_time;
        document.getElementById('total-results').textContent = data.results.length;

        currentResults = data.results;
        currentPage = 1;
        renderTable();
        renderPagination();

        // Display trace visualization
        renderProfile(data.profile_data);
    }

    function displayAnalyseResults(data) {
        document.getElementById('results-card').style.display = 'block';
        // document.getElementById('results-info').style.display = 'none';
        // document.getElementById('results-table-container').style.display = 'none';
        // document.getElementById('pagination-controls').style.display = 'none';
        // document.getElementById('profile-chart').style.display = 'none';
        
        const analyseChart = document.getElementById('analyse-chart');
        analyseChart.style.display = 'block';
        const recallChart = document.getElementById('recall-chart');
        recallChart.style.display = 'block';
        
        renderAnalyseChart(data.series);
        if (data.recall) {
            renderRecallChart(data.recall);
        }
    }

    function renderRecallChart(recallData) {
        const chartDom = document.getElementById('recall-chart');
        let myChart = echarts.getInstanceByDom(chartDom);
        if (myChart) {
            myChart.dispose();
        }
        myChart = echarts.init(chartDom, null, {renderer: 'svg'});

        const option = {
            title: {
                text: 'Recall vs Sparsity',
                left: 'center',
                textStyle: { fontSize: "1em" }
            },
            tooltip: {
                trigger: 'axis'
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: 30,
                containLabel: true
            },
            xAxis: {
                type: 'category',
                name: 'Sparsity',
                nameLocation: 'middle',
                nameGap: 30,
                nameTextStyle: { fontSize: "0.97em" },
                data: recallData.map(item => item.sparsity),
                // axisTick: {
                //     alignWithLabel: true
                // },
                axisLabel: { fontSize: "0.95em" }
            },
            yAxis: {
                type: 'value',
                name: 'Recall',
                nameTextStyle: { fontSize: "0.97em" },
                min: 0,
                max: 1,
                axisLabel: { fontSize: "0.95em" }
            },
            series: [
                {
                    name: 'Recall',
                    type: 'line',
                    data: recallData.map(item => item.recall),
                    smooth: true,
                    itemStyle: {
                        color: '#ee6666'
                    },
                    label: {
                        show: true,
                        position: 'top',
                        fontSize: "0.97em",
                        formatter: function(params) {
                            return params.value.toFixed(2);
                        }
                    }
                }
            ]
        };        myChart.setOption(option);
        
        window.addEventListener('resize', function() {
            myChart.resize();
        });
    }

    function renderAnalyseChart(seriesData) {
        const chartDom = document.getElementById('analyse-chart');
        let myChart = echarts.getInstanceByDom(chartDom);
        if (myChart) {
            myChart.dispose();
        }
        myChart = echarts.init(chartDom, null, {renderer: 'svg'});

        const option = {
            title: {
                text: 'Inference Time Comparison',
                left: 'center',
                textStyle: { fontSize: "1em" }
            },
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'shadow'
                }
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                data: seriesData.map(item => item.name),
                axisTick: {
                    alignWithLabel: true
                },
                axisLabel: { fontSize: "0.97em" }
            },
            yAxis: {
                type: 'value',
                name: 'Time (s)',
                nameTextStyle: { fontSize: "0.97em" },
                axisLabel: { fontSize: "0.95em" }
            },
            series: [
                {
                    name: 'Time',
                    type: 'bar',
                    barWidth: '60%',
                    data: seriesData.map(item => item.value),
                    itemStyle: {
                        color: function(params) {
                            const colors = ['#5470c6', '#91cc75', '#fac858'];
                            return colors[params.dataIndex % colors.length];
                        }
                    },
                    label: {
                        show: true,
                        position: 'top',
                        fontSize: "0.97em",
                        formatter: function(params) {
                            return params.value.toFixed(2) + ' s';
                        }
                    }
                }
            ]
        };

        myChart.setOption(option);
        
        window.addEventListener('resize', function() {
            myChart.resize();
        });
    }

    function renderTable() {
        const tableHead = document.querySelector('#results-table thead');
        const tableBody = document.querySelector('#results-table tbody');
        tableHead.innerHTML = '';
        tableBody.innerHTML = '';

        if (currentResults.length > 0) {
            const headers = Object.keys(currentResults[0]);
            const headerRow = document.createElement('tr');
            headers.forEach(header => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            tableHead.appendChild(headerRow);

            const start = (currentPage - 1) * rowsPerPage;
            const end = start + rowsPerPage;
            const pageData = currentResults.slice(start, end);

            pageData.forEach(row => {
                const tr = document.createElement('tr');
                headers.forEach(header => {
                    const td = document.createElement('td');
                    td.textContent = row[header];
                    td.title = row[header]; // Tooltip for full text
                    tr.appendChild(td);
                });
                tableBody.appendChild(tr);
            });
        }
    }

    function renderPagination() {
        const paginationControls = document.getElementById('pagination-controls');
        if (currentResults.length <= rowsPerPage) {
            paginationControls.style.display = 'none';
            return;
        }

        paginationControls.style.display = 'block';
        const totalPages = Math.ceil(currentResults.length / rowsPerPage);
        document.getElementById('page-info').textContent = `Page ${currentPage} of ${totalPages}`;
        
        document.getElementById('prev-page').disabled = currentPage === 1;
        document.getElementById('next-page').disabled = currentPage === totalPages;
    }

    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderTable();
            renderPagination();
        }
    });

    document.getElementById('next-page').addEventListener('click', () => {
        const totalPages = Math.ceil(currentResults.length / rowsPerPage);
        if (currentPage < totalPages) {
            currentPage++;
            renderTable();
            renderPagination();
        }
    });

    function renderProfile(profileData) {
        const chartDom = document.getElementById('profile-chart');
        let myChart = echarts.getInstanceByDom(chartDom);
        if (myChart) {
            myChart.dispose();
        }
        myChart = echarts.init(chartDom, null, {renderer: 'svg'});
        
        if (!profileData || profileData.length === 0) {
            chartDom.innerHTML = '<p style="text-align:center; padding-top: 20px; color: #7f8c8d;">No profile data available.</p>';
            return;
        }

        const categories = ['Transfer', 'Inference'];

        function renderItem(params, api) {
            var categoryIndex = api.value(0);
            var start = api.coord([api.value(1), categoryIndex]);
            var end = api.coord([api.value(2), categoryIndex]);
            var height = api.size([0, 1])[1] * 0.6;
            var rectShape = echarts.graphic.clipRectByRect(
                {
                    x: start[0],
                    y: start[1] - height / 2,
                    width: end[0] - start[0],
                    height: height
                },
                {
                    x: params.coordSys.x,
                    y: params.coordSys.y,
                    width: params.coordSys.width,
                    height: params.coordSys.height
                }
            );
            return (
                rectShape && {
                    type: 'rect',
                    transition: ['shape'],
                    shape: rectShape,
                    style: api.style()
                }
            );
        }

        const option = {
            tooltip: {
                formatter: function (params) {
                    return params.marker + params.name + ': ' + params.value[3].toFixed(2) + ' ms';
                }
            },
            title: {
                text: 'Execution Profile',
                left: 'center',
                textStyle: { fontSize: "1em" }
            },
            dataZoom: [
                {
                    type: 'slider',
                    filterMode: 'weakFilter',
                    showDataShadow: false,
                    top: 260,
                    labelFormatter: ''
                },
                {
                    type: 'inside',
                    filterMode: 'weakFilter'
                }
            ],
            grid: {
                height: 180,
                top: 40
            },
            xAxis: {
                min: 0,
                scale: true,
                axisLabel: {
                    fontSize: "0.95em",
                    formatter: function (val) {
                        return val + ' ms';
                    }
                }
            },
            yAxis: {
                data: categories,
                inverse: true,
                axisLabel: { fontSize: "0.97em" }
            },
            series: [
                {
                    type: 'custom',
                    renderItem: renderItem,
                    itemStyle: {
                        opacity: 0.8
                    },
                    encode: {
                        x: [1, 2],
                        y: 0
                    },
                    data: profileData
                }
            ]
        };

        myChart.setOption(option);
        
        // Handle resize
        window.addEventListener('resize', function() {
            myChart.resize();
        });
    }
});
