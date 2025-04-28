/**
 * Visualization script for LLM Error Correction Activation Analysis
 */

let resultData = null;
let currentLayer = null;
let selectedSentenceTypes = [];
let selectedPairId = null;
let sortOrder = 'index';

const resultFileSelect = document.getElementById('resultFile');
const layerSelect = document.getElementById('layerSelect');
const sentenceTypeCheckboxes = document.getElementById('sentenceTypeCheckboxes');
const pairSelect = document.getElementById('pairSelect');
const sortOrderSelect = document.getElementById('sortOrder');
const plotContainer = document.getElementById('plotContainer');
const controlPanel = document.getElementById('controlPanel');
const plotTitle = document.getElementById('plotTitle');
const sentencePairInfo = document.getElementById('sentencePairInfo');
const sentence1Element = document.getElementById('sentence1');
const sentence2Element = document.getElementById('sentence2');
const loadingElement = document.querySelector('.loading');

document.addEventListener('DOMContentLoaded', () => {
    fetchResultFiles();
    
    resultFileSelect.addEventListener('change', handleResultFileChange);
    layerSelect.addEventListener('change', handleLayerChange);
    pairSelect.addEventListener('change', handlePairChange);
    sortOrderSelect.addEventListener('change', handleSortOrderChange);
});

/**
 * Fetch available result files from the server
 */
function fetchResultFiles() {
    showLoading(true);
    
    fetch('/list_results')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading result files:', data.error);
                alert('結果ファイルの読み込みエラー: ' + data.error);
                return;
            }
            
            populateResultFileSelect(data.results);
        })
        .catch(error => {
            console.error('Error fetching result files:', error);
            alert('結果ファイルの取得エラー: ' + error.message);
        })
        .finally(() => {
            showLoading(false);
        });
}

/**
 * Populate the result file select dropdown
 */
function populateResultFileSelect(results) {
    resultFileSelect.innerHTML = '<option value="">ファイルを選択してください</option>';
    
    if (results.length === 0) {
        const option = document.createElement('option');
        option.disabled = true;
        option.textContent = '利用可能な結果ファイルがありません';
        resultFileSelect.appendChild(option);
        return;
    }
    
    results.forEach(result => {
        const option = document.createElement('option');
        option.value = result.full_path;
        option.textContent = result.path;
        resultFileSelect.appendChild(option);
    });
}

/**
 * Handle result file selection change
 */
function handleResultFileChange() {
    const selectedFile = resultFileSelect.value;
    
    if (!selectedFile) {
        resetUI();
        return;
    }
    
    showLoading(true);
    
    fetch(`/load_result?path=${encodeURIComponent(selectedFile)}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error loading result file:', data.error);
                alert('結果ファイルの読み込みエラー: ' + data.error);
                resetUI();
                return;
            }
            
            resultData = data.data;
            
            populateLayerSelect(data.metadata.layers);
            populateSentenceTypeCheckboxes(data.metadata.sentence_types);
            
            controlPanel.style.display = 'block';
        })
        .catch(error => {
            console.error('Error fetching result file:', error);
            alert('結果ファイルの取得エラー: ' + error.message);
            resetUI();
        })
        .finally(() => {
            showLoading(false);
        });
}

/**
 * Populate the layer select dropdown
 */
function populateLayerSelect(layers) {
    layerSelect.innerHTML = '<option value="">レイヤーを選択してください</option>';
    
    layers.forEach(layer => {
        const option = document.createElement('option');
        option.value = layer;
        option.textContent = `レイヤー ${layer.split('_')[1]}`;
        layerSelect.appendChild(option);
    });
}

/**
 * Populate the sentence type checkboxes
 */
function populateSentenceTypeCheckboxes(sentenceTypes) {
    sentenceTypeCheckboxes.innerHTML = '';
    selectedSentenceTypes = [];
    
    sentenceTypes.forEach(type => {
        const checkboxId = `sentenceType_${type}`;
        
        const checkboxDiv = document.createElement('div');
        checkboxDiv.className = 'form-check';
        
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.className = 'form-check-input';
        checkbox.id = checkboxId;
        checkbox.value = type;
        checkbox.checked = true;  // Default to checked
        selectedSentenceTypes.push(type);
        
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) {
                selectedSentenceTypes.push(type);
            } else {
                selectedSentenceTypes = selectedSentenceTypes.filter(t => t !== type);
            }
            updatePairSelect();
            updateVisualization();
        });
        
        const label = document.createElement('label');
        label.className = 'form-check-label';
        label.htmlFor = checkboxId;
        label.textContent = type;
        
        checkboxDiv.appendChild(checkbox);
        checkboxDiv.appendChild(label);
        sentenceTypeCheckboxes.appendChild(checkboxDiv);
    });
}

/**
 * Handle layer selection change
 */
function handleLayerChange() {
    currentLayer = layerSelect.value;
    
    if (!currentLayer) {
        plotContainer.style.display = 'none';
        plotTitle.style.display = 'none';
        return;
    }
    
    updatePairSelect();
    updateVisualization();
}

/**
 * Update the pair select dropdown based on selected layer and sentence types
 */
function updatePairSelect() {
    pairSelect.innerHTML = '<option value="">文ペアを選択してください</option>';
    
    if (!currentLayer || selectedSentenceTypes.length === 0) {
        return;
    }
    
    const pairs = [];
    
    selectedSentenceTypes.forEach(type => {
        const subtypes = resultData.diff_token_type[type] || {};
        
        Object.entries(subtypes).forEach(([subtype, pairData]) => {
            Object.entries(pairData).forEach(([pairId, data]) => {
                if (data.layers && data.layers[currentLayer]) {
                    pairs.push({
                        id: pairId,
                        type: type,
                        subtype: subtype,
                        sentence1: data.sentence_pair.sentence1,
                        sentence2: data.sentence_pair.sentence2
                    });
                }
            });
        });
    });
    
    pairs.forEach(pair => {
        const option = document.createElement('option');
        option.value = JSON.stringify({
            id: pair.id,
            type: pair.type,
            subtype: pair.subtype
        });
        option.textContent = `${pair.type} / ${pair.subtype} / ${pair.id}`;
        pairSelect.appendChild(option);
    });
    
    if (pairs.length > 0) {
        pairSelect.selectedIndex = 1;
        handlePairChange();
    }
}

/**
 * Handle pair selection change
 */
function handlePairChange() {
    if (!pairSelect.value) {
        sentencePairInfo.style.display = 'none';
        return;
    }
    
    const pairInfo = JSON.parse(pairSelect.value);
    selectedPairId = pairInfo;
    
    const pairData = resultData.diff_token_type[pairInfo.type][pairInfo.subtype][pairInfo.id];
    sentence1Element.textContent = pairData.sentence_pair.sentence1;
    sentence2Element.textContent = pairData.sentence_pair.sentence2;
    sentencePairInfo.style.display = 'block';
    
    updateVisualization();
}

/**
 * Handle sort order change
 */
function handleSortOrderChange() {
    sortOrder = sortOrderSelect.value;
    updateVisualization();
}

/**
 * Update the visualization based on current selections
 */
function updateVisualization() {
    if (!currentLayer || !selectedPairId) {
        return;
    }
    
    const pairInfo = selectedPairId;
    const pairData = resultData.diff_token_type[pairInfo.type][pairInfo.subtype][pairInfo.id];
    const layerData = pairData.layers[currentLayer];
    
    if (!layerData || !layerData.dimensions) {
        console.error('No dimension data found for selected layer');
        return;
    }
    
    const dimensions = Object.entries(layerData.dimensions).map(([dimName, dimData]) => {
        return {
            dimension: dimName,
            dimensionIndex: parseInt(dimName.split('_')[1]),
            activationDiff: dimData.activation_diff,
            sentence1Activation: dimData.sentence1_activation,
            sentence2Activation: dimData.sentence2_activation
        };
    });
    
    if (sortOrder === 'ascending') {
        dimensions.sort((a, b) => a.activationDiff - b.activationDiff);
    } else if (sortOrder === 'descending') {
        dimensions.sort((a, b) => b.activationDiff - a.activationDiff);
    } else {
        dimensions.sort((a, b) => a.dimensionIndex - b.dimensionIndex);
    }
    
    const trace = {
        x: dimensions.map(d => d.dimensionIndex),
        y: dimensions.map(d => d.activationDiff),
        mode: 'lines+markers',
        type: 'scatter',
        name: `${pairInfo.type} / ${pairInfo.subtype}`,
        hovertemplate: 
            'Dimension: %{x}<br>' +
            'Activation Diff: %{y:.6f}<br>' +
            '<extra></extra>'
    };
    
    const layout = {
        title: `レイヤー ${currentLayer.split('_')[1]} の活性化差分`,
        xaxis: {
            title: 'Dimension',
            tickmode: 'auto',
            nticks: 20
        },
        yaxis: {
            title: 'Activation Difference',
            rangemode: 'tozero'
        },
        hovermode: 'closest',
        showlegend: true
    };
    
    Plotly.newPlot(plotContainer, [trace], layout);
    
    plotContainer.style.display = 'block';
    plotTitle.style.display = 'block';
    plotTitle.textContent = `レイヤー ${currentLayer.split('_')[1]} の活性化差分`;
}

/**
 * Reset the UI to its initial state
 */
function resetUI() {
    resultData = null;
    currentLayer = null;
    selectedSentenceTypes = [];
    selectedPairId = null;
    
    controlPanel.style.display = 'none';
    plotContainer.style.display = 'none';
    plotTitle.style.display = 'none';
    sentencePairInfo.style.display = 'none';
}

/**
 * Show or hide the loading indicator
 */
function showLoading(show) {
    loadingElement.style.display = show ? 'block' : 'none';
}
