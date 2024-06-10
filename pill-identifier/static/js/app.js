let pillDetails = [];

document.addEventListener('DOMContentLoaded', function() {
    const frontImageUpload = document.getElementById('front-image-upload');
    const backImageUpload = document.getElementById('back-image-upload');
    const previewFrontImage = document.getElementById('preview-front-image');
    const previewBackImage = document.getElementById('preview-back-image');
    const identifyBtn = document.getElementById('identify-btn');
    const backToIdentifyBtn = document.getElementById('back-to-identify');
    const backToResultBtn = document.getElementById('back-to-result');
    const identifyPage = document.getElementById('identify-page');
    const resultPage = document.getElementById('result-page');
    const detailPage = document.getElementById('detail-page');
    const loadingPage = document.getElementById('loading-page');
    const pillImagesDiv = document.getElementById('pill-images');
    const pillNameElement = document.getElementById('pill-name');
    const efficacyElement = document.getElementById('efficacy');
    const dosageElement = document.getElementById('dosage');
    const precautionsElement = document.getElementById('precautions');
    const detailImage = document.getElementById('detail-image');

    function previewImage(event, previewElement) {
        const file = event.target.files[0];
        const reader = new FileReader();

        reader.onload = e => {
            previewElement.src = e.target.result;
            previewElement.style.display = 'block';
        };

        reader.readAsDataURL(file);
    }

    function displayResults(data) {
        pillDetails = data;
    
        pillImagesDiv.innerHTML = pillDetails.map(pill => `
            <div class="pill-item">
                <img src="${pill['이미지']}" alt="${pill['품목명']}" class="mb-2">
                <button class="pill-name-btn bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" data-id="${pill['품목일련번호']}">
                    ${pill['품목명']}
                </button>
            </div>
        `).join('');
    
        pillImagesDiv.addEventListener('click', handlePillClick);
    }

    function handlePillClick(event) {
        const btn = event.target.closest('.pill-name-btn');
        if (!btn) return;

        const pillId = parseInt(btn.dataset.id, 10);
        const selectedPill = pillDetails.find(pill => pill['품목일련번호'] === pillId);

        if (selectedPill) {
            pillNameElement.textContent = selectedPill['품목명'] || '품목명 정보 없음';
            efficacyElement.textContent = selectedPill['효능효과'] || '효능효과 정보 없음';
            dosageElement.textContent = selectedPill['용법용량'] || '용법용량 정보 없음';
            precautionsElement.textContent = selectedPill['주의사항'] || '주의사항 정보 없음';

            detailImage.src = selectedPill['이미지'];
            detailImage.style.display = "block";
            resultPage.classList.add('hidden');
            detailPage.classList.remove('hidden');
        } else {
            alert('선택한 약품 정보를 찾을 수 없습니다.');
        }
    }

    function handleIdentify() {
        const frontFile = frontImageUpload.files[0];
        const backFile = backImageUpload.files[0];

        if (!frontFile || !backFile) {
            alert('앞면과 뒷면 이미지를 모두 업로드해주세요.');
            return;
        }

        identifyPage.classList.add('hidden');
        loadingPage.classList.remove('hidden');

        const url = 'http://172.30.1.48:8080/identify';
        const formData = new FormData();
        formData.append('front_image', frontFile);
        formData.append('back_image', backFile);

        fetch(url, {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) throw new Error('서버에서 응답을 받지 못했습니다.');
            return response.json();
        })
        .then(data => {
            displayResults(data.pill_details); // 수정된 부분
            loadingPage.classList.add('hidden');
            resultPage.classList.remove('hidden');
        })
        .catch(error => {
            console.error('알약 식별 중 에러 발생:', error);
            resultPage.innerHTML = '<p>알약 식별 중 오류가 발생했습니다.</p>';
            loadingPage.classList.add('hidden');
            resultPage.classList.remove('hidden');
        });
    }

    function handleBackToIdentify() {
        identifyPage.classList.remove('hidden');
        resultPage.classList.add('hidden');
        detailPage.classList.add('hidden');
    }

    function handleBackToResult() {
        resultPage.classList.remove('hidden');
        detailPage.classList.add('hidden');
    }

    frontImageUpload.addEventListener('change', e => previewImage(e, previewFrontImage));
    backImageUpload.addEventListener('change', e => previewImage(e, previewBackImage));
    identifyBtn.addEventListener('click', handleIdentify);
    backToIdentifyBtn.addEventListener('click', handleBackToIdentify);
    backToResultBtn.addEventListener('click', handleBackToResult);
});