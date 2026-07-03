document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector("form");
    const button = document.querySelector(".predict-btn");

    if (form && button) {
        form.addEventListener("submit", function () {
            button.innerText = "Predicting...";
            button.disabled = true;
        });
    }

    const riskText = document.querySelector(".risk");

    if (riskText) {
        const value = riskText.innerText.toLowerCase();

        if (value.includes("low")) {
            riskText.style.background = "#22c55e";
        } else if (value.includes("medium")) {
            riskText.style.background = "#f59e0b";
        } else {
            riskText.style.background = "#ef4444";
        }
    }
});