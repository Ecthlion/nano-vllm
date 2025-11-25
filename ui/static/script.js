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

        const progressBar = document.getElementById('upload-progress');
        const progressContainer = document.getElementById('upload-progress-container');
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
            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            statusP.textContent = 'Uploading...';

            let width = 0;
            const interval = setInterval(() => {
                width += 5; // Fast simulation
                progressBar.style.width = width + '%';
                if (width >= 100) {
                    clearInterval(interval);
                    statusP.textContent = `File uploaded successfully: ${file.name}`;
                    statusP.style.color = 'green';
                    uploadedFilepath = filepath;
                    document.getElementById('index-card').style.display = 'block';
                }
            }, 20); // Fast interval
        }

        function performUpload(file) {
            const formData = new FormData();
            formData.append('file', file);

            progressContainer.style.display = 'block';
            progressBar.style.width = '0%';
            statusP.textContent = 'Uploading...';

            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', function(e) {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    progressBar.style.width = percentComplete + '%';
                }
            });

            xhr.onload = function() {
                if (xhr.status === 200) {
                    const data = JSON.parse(xhr.responseText);
                    progressBar.style.width = '100%';
                    if (data.error) {
                        statusP.textContent = `Error: ${data.error}`;
                        statusP.style.color = 'red';
                    } else {
                        statusP.textContent = `File uploaded successfully: ${file.name}`;
                        statusP.style.color = 'green';
                        uploadedFilepath = data.filepath;
                        document.getElementById('index-card').style.display = 'block';
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

        const progressBar = document.getElementById('index-progress');
        const progressContainer = document.getElementById('index-progress-container');
        const indexInfo = document.getElementById('index-info');

        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        indexInfo.style.display = 'none';

        // Simulate index building progress
        let width = 0;
        const interval = setInterval(() => {
            width += 5;
            progressBar.style.width = width + '%';
            if (width >= 100) {
                clearInterval(interval);
            }
        }, 150);

        fetch('/build_index', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filepath: uploadedFilepath, sparsity: sparsity, field: field })
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            progressBar.style.width = '100%';
            if (data.error) {
                alert(`Error: ${data.error}`);
            } else {
                document.getElementById('index-size').textContent = data.index_size;
                indexInfo.style.display = 'block';
                // document.getElementById('query-card').style.display = 'block'; // Already visible
            }
        })
        .catch(error => {
            clearInterval(interval);
            alert(`Error: ${error.message}`);
        });
    });

    // --- Step 3: Query ---
    document.getElementById('query-with-index').addEventListener('click', () => runQuery(true));
    document.getElementById('query-without-index').addEventListener('click', () => runQuery(false));

    function runQuery(useIndex) {
        const query = document.getElementById('query-input').value;
        if (!query) {
            alert('Please enter a query.');
            return;
        }

        const progressBar = document.getElementById('query-progress');
        const progressContainer = document.getElementById('query-progress-container');
        
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        document.getElementById('results-card').style.display = 'none';

        // Simulate query progress
        let width = 0;
        const interval = setInterval(() => {
            width += 10;
            progressBar.style.width = width + '%';
            if (width >= 100) {
                clearInterval(interval);
            }
        }, 200);

        fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, use_index: useIndex })
        })
        .then(response => response.json())
        .then(data => {
            clearInterval(interval);
            progressBar.style.width = '100%';
            displayResults(data);
        })
        .catch(error => {
            clearInterval(interval);
            alert(`Error: ${error.message}`);
        });
    }

    // --- Step 4: Display Results ---
    let currentResults = [];
    let currentPage = 1;
    const rowsPerPage = 10;

    function displayResults(data) {
        document.getElementById('results-card').style.display = 'block';
        
        // Display inference time
        document.getElementById('inference-time').textContent = data.inference_time;
        document.getElementById('total-results').textContent = data.results.length;

        currentResults = data.results;
        currentPage = 1;
        renderTable();
        renderPagination();

        // Display trace visualization
        renderTrace(data.trace_data);
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

    function renderTrace(traceData) {
        const container = document.getElementById('trace-container');
        container.innerHTML = '';

        if (!traceData || !traceData.traceEvents || traceData.traceEvents.length === 0) {
            container.innerHTML = '<p style="text-align:center; padding-top: 20px; color: #7f8c8d;">No trace data available.</p>';
            return;
        }

        const margin = { top: 20, right: 30, bottom: 30, left: 90 };
        const width = container.clientWidth - margin.left - margin.right;
        const height = container.clientHeight - margin.top - margin.bottom;

        const svg = d3.select(container)
            .append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin.left},${margin.top})`);

        const events = traceData.traceEvents;
        const tids = [...new Set(events.map(d => d.tid))];
        
        const x = d3.scaleLinear()
            .domain([0, d3.max(events, d => d.ts + d.dur)])
            .range([0, width]);

        const y = d3.scaleBand()
            .domain(tids)
            .range([0, height])
            .padding(0.1);

        const color = d3.scaleOrdinal(d3.schemeCategory10).domain(tids);

        svg.append("g")
            .attr("transform", `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(width / 80).tickFormat(d => `${d/1000}ms`));

        svg.append("g")
            .call(d3.axisLeft(y).tickFormat(d => `Thread ${d}`));

        svg.selectAll("rect")
            .data(events)
            .enter()
            .append("rect")
            .attr("x", d => x(d.ts))
            .attr("y", d => y(d.tid))
            .attr("width", d => x(d.dur))
            .attr("height", y.bandwidth())
            .attr("fill", d => color(d.tid));
        
        svg.selectAll(".text")
            .data(events)
            .enter()
            .append("text")
            .attr("x", d => x(d.ts) + 5)
            .attr("y", d => y(d.tid) + y.bandwidth() / 2)
            .attr("dy", ".35em")
            .attr("fill", "white")
            .style("font-size", "10px")
            .text(d => d.name);
    }
});
