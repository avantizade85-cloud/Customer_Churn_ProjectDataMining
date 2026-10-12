// =====================================================
// CUSTOMER CHURN ANALYTICS - COMPLETE JAVASCRIPT
// =====================================================


// =====================================================
// 1. PAGE NAVIGATION
// =====================================================

function showSection(sectionId, clickedButton) {

    document.querySelectorAll(".page-section").forEach(section => {
        section.classList.remove("active-section");
    });

    const section = document.getElementById(sectionId);

    if (section) {
        section.classList.add("active-section");
    }

    document.querySelectorAll(".nav-item").forEach(item => {
        item.classList.remove("active");
    });

    if (clickedButton) {
        clickedButton.classList.add("active");
    }

    updateHeader(sectionId);
}


// =====================================================
// 2. HEADER TITLES
// =====================================================

function updateHeader(sectionId) {

    const title = document.getElementById("page-title");
    const subtitle = document.getElementById("page-subtitle");

    const pages = {

        dashboard: [
            "Customer Churn Dashboard",
            "Analyze customer behavior and identify churn risk."
        ],

        analytics: [
            "Customer Data Analysis",
            "Explore important customer churn patterns."
        ],

        segmentation: [
            "Customer Segmentation",
            "Discover customer groups using K-Means clustering."
        ],

        prediction: [
            "Customer Churn Prediction",
            "Use machine learning to estimate customer churn risk."
        ],

        "dataset-prediction": [
            "Dataset-Based Churn Prediction",
            "Run churn predictions on your customer dataset."
        ],

        performance: [
            "Model Performance",
            "Compare machine learning models used for churn prediction."
        ],

        recommendations: [
            "Business Recommendations",
            "Recommended actions based on customer churn risk."
        ]
    };

    if (title && subtitle && pages[sectionId]) {

        title.textContent = pages[sectionId][0];

        subtitle.textContent = pages[sectionId][1];
    }
}


// =====================================================
// 3. HELPER FUNCTION
// =====================================================

function setText(id, value) {

    const element = document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}


// =====================================================
// 4. LOAD DASHBOARD
// =====================================================

async function loadDashboard() {

    try {

        const response = await fetch("/api/dashboard");

        const data = await response.json();

        if (!response.ok || data.success === false) {

            throw new Error(
                data.error || "Could not load dashboard data."
            );
        }

        setText(
            "total-customers",
            Number(data.total_customers).toLocaleString()
        );

        setText(
            "churned-customers",
            Number(data.churned_customers).toLocaleString()
        );

        setText(
            "churn-rate",
            data.churn_rate + "%"
        );

        setText(
            "avg-charges",
            "₹ " + data.average_monthly_charges
        );

    } catch (error) {

        console.error("Dashboard error:", error);

        setText("total-customers", "Unavailable");
        setText("churned-customers", "Unavailable");
        setText("churn-rate", "Unavailable");
        setText("avg-charges", "Unavailable");
    }
}


// =====================================================
// 5. OPEN INDIVIDUAL PREDICTION
// =====================================================

function activatePrediction() {

    const button = Array.from(
        document.querySelectorAll(".nav-item")
    ).find(item =>
        (item.getAttribute("onclick") || "")
            .includes("showSection('prediction'")
    );

    showSection("prediction", button);
}


// =====================================================
// 6. INDIVIDUAL CUSTOMER PREDICTION
// =====================================================

async function handleIndividualPrediction(event) {

    event.preventDefault();

    const form = document.getElementById("prediction-form");

    const button = form
        ? form.querySelector('button[type="submit"]')
        : null;

    if (!form || !button) {
        return;
    }

    const originalText = button.textContent;

    button.disabled = true;

    button.textContent = "⏳ Predicting...";

    const getValue = id => {

        const element = document.getElementById(id);

        if (!element) {
            throw new Error("Missing form field: " + id);
        }

        return element.value;
    };

    try {

        const customerData = {

            gender: getValue("gender"),

            SeniorCitizen: getValue("SeniorCitizen"),

            Partner: getValue("Partner"),

            Dependents: getValue("Dependents"),

            tenure: getValue("tenure"),

            PhoneService: getValue("PhoneService"),

            MultipleLines: getValue("MultipleLines"),

            InternetService: getValue("InternetService"),

            OnlineSecurity: getValue("OnlineSecurity"),

            OnlineBackup: getValue("OnlineBackup"),

            DeviceProtection: getValue("DeviceProtection"),

            TechSupport: getValue("TechSupport"),

            StreamingTV: getValue("StreamingTV"),

            StreamingMovies: getValue("StreamingMovies"),

            Contract: getValue("Contract"),

            PaperlessBilling: getValue("PaperlessBilling"),

            PaymentMethod: getValue("PaymentMethod"),

            MonthlyCharges: getValue("MonthlyCharges"),

            TotalCharges: getValue("TotalCharges")
        };

        console.log(
            "Sending customer data:",
            customerData
        );

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(customerData)

        });

        const result = await response.json();

        console.log(
            "Prediction response:",
            result
        );

        if (!response.ok || !result.success) {

            throw new Error(
                result.error || "Prediction failed."
            );
        }

        const placeholder = document.getElementById(
            "prediction-placeholder"
        );

        const resultPanel = document.getElementById(
            "prediction-result"
        );

        const badge = document.getElementById(
            "prediction-badge"
        );

        if (placeholder) {
            placeholder.classList.add("hidden");
        }

        if (resultPanel) {
            resultPanel.classList.remove("hidden");
        }

        if (badge) {

            if (result.prediction === "Yes") {

                badge.textContent =
                    "⚠️ CUSTOMER PREDICTED TO CHURN";

                badge.className =
                    "prediction-badge churn";

            } else {

                badge.textContent =
                    "✅ CUSTOMER PREDICTED NOT TO CHURN";

                badge.className =
                    "prediction-badge no-churn";
            }
        }

        setText(
            "probability-value",
            Number(result.probability).toFixed(2) + "%"
        );

        setText(
            "risk-value",
            result.risk_level
        );

        setText(
            "recommendation-value",
            result.recommendation
        );

    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        alert(
            "Prediction failed:\n" + error.message
        );

    } finally {

        button.disabled = false;

        button.textContent =
            originalText || "🔮 Predict Customer Churn";
    }
}


// =====================================================
// 7. CUSTOMER SEGMENTATION USING K-MEANS
// =====================================================

async function findSegment() {

    const tenure = document.getElementById(
        "segment-tenure"
    )?.value;

    const monthly = document.getElementById(
        "segment-monthly"
    )?.value;

    const total = document.getElementById(
        "segment-total"
    )?.value;

    const resultBox = document.getElementById(
        "segment-result"
    );

    if (!resultBox) {
        return;
    }

    if (
        tenure === "" ||
        monthly === "" ||
        total === "" ||
        tenure == null ||
        monthly == null ||
        total == null
    ) {

        resultBox.classList.remove("hidden");

        resultBox.textContent =
            "Please enter all customer values.";

        return;
    }

    try {

        resultBox.classList.remove("hidden");

        resultBox.textContent =
            "⏳ Finding customer segment...";

        const response = await fetch("/segment", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                tenure: tenure,

                MonthlyCharges: monthly,

                TotalCharges: total
            })

        });

        const result = await response.json();

        if (!response.ok || !result.success) {

            throw new Error(
                result.error || "Segmentation failed."
            );
        }

        const segmentNumber = Number(result.segment) + 1;

        resultBox.innerHTML = `

            <div style="
                padding:18px;
                background:#eef2ff;
                border-radius:12px;
                margin-top:15px;
            ">

                <div style="
                    font-size:12px;
                    color:#697386;
                    margin-bottom:7px;
                ">
                    CUSTOMER SEGMENT
                </div>

                <div style="
                    font-size:24px;
                    font-weight:800;
                    color:#4f46e5;
                ">
                    Customer Segment ${segmentNumber}
                </div>

                <div style="
                    margin-top:8px;
                    font-size:13px;
                    color:#697386;
                ">
                    Cluster ID: ${Number(result.segment)}
                </div>

            </div>
        `;

    } catch (error) {

        console.error(
            "Segmentation error:",
            error
        );

        resultBox.textContent =
            "Segmentation failed: " + error.message;
    }
}


// =====================================================
// 8. MODEL PERFORMANCE
// =====================================================

async function loadModelInfo() {

    try {

        const response = await fetch("/api/model-info");

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.error || "Model information unavailable."
            );
        }

        setText(
            "best-model-name",
            data["Best Model"] || "Unavailable"
        );

        setText(
            "model-accuracy",
            (Number(data["Accuracy"]) * 100).toFixed(2) + "%"
        );

        setText(
            "model-precision",
            (Number(data["Precision"]) * 100).toFixed(2) + "%"
        );

        setText(
            "model-recall",
            (Number(data["Recall"]) * 100).toFixed(2) + "%"
        );

        setText(
            "model-f1",
            (Number(data["F1 Score"]) * 100).toFixed(2) + "%"
        );

        setText(
            "model-roc",
            (Number(data["ROC-AUC"]) * 100).toFixed(2) + "%"
        );

    } catch (error) {

        console.error(
            "Model information error:",
            error
        );
    }
}


// =====================================================
// 9. DATASET-BASED CHURN PREDICTION
// =====================================================

async function runDatasetPrediction() {

    const button = document.getElementById(
        "run-dataset-button"
    );

    const status = document.getElementById(
        "dataset-status"
    );

    const results = document.getElementById(
        "dataset-results"
    );

    if (!button || !status || !results) {

        console.error(
            "Dataset prediction page elements were not found. Check index.html."
        );

        return;
    }

    const originalText = button.textContent;

    button.disabled = true;

    button.textContent =
        "⏳ Processing Dataset...";

    status.textContent =
        "Loading dataset and generating predictions. Please wait...";

    status.style.color = "#4f46e5";

    results.classList.add("hidden");

    try {

        // Call the Flask dataset-prediction endpoint.
        const response = await fetch(
            "/predict-dataset",
            {
                method: "POST"
            }
        );

        const data = await response.json();

        if (!response.ok || !data.success) {

            throw new Error(
                data.error || "Dataset prediction failed."
            );
        }

        // -----------------------------------------
        // Display dataset prediction summary
        // -----------------------------------------

        setText(
            "rows-processed",
            Number(data.rows_processed).toLocaleString()
        );

        setText(
            "predicted-churn",
            Number(data.predicted_churn).toLocaleString()
        );

        setText(
            "predicted-no-churn",
            Number(data.predicted_no_churn).toLocaleString()
        );

        setText(
            "high-risk-count",
            Number(data.high_risk_customers).toLocaleString()
        );

        setText(
            "medium-risk-count",
            Number(data.medium_risk_customers).toLocaleString()
        );

        setText(
            "low-risk-count",
            Number(data.low_risk_customers).toLocaleString()
        );

        // -----------------------------------------
        // Display held-out test metrics
        // -----------------------------------------

        const metrics = data.test_metrics || {};

        setText(
            "test-accuracy",
            metrics.accuracy == null
                ? "-"
                : metrics.accuracy + "%"
        );

        setText(
            "test-precision",
            metrics.precision == null
                ? "-"
                : metrics.precision + "%"
        );

        setText(
            "test-recall",
            metrics.recall == null
                ? "-"
                : metrics.recall + "%"
        );

        setText(
            "test-f1",
            metrics.f1_score == null
                ? "-"
                : metrics.f1_score + "%"
        );

        setText(
            "test-roc-auc",
            metrics.roc_auc == null
                ? "-"
                : metrics.roc_auc + "%"
        );

        // -----------------------------------------
        // Configure the CSV download button
        // -----------------------------------------

        const downloadLink = document.getElementById(
            "dataset-download-link"
        );

        if (downloadLink) {

            downloadLink.href =
                data.download_url || "/download-predictions";
        }

        // -----------------------------------------
        // Show results
        // -----------------------------------------

        results.classList.remove("hidden");

        status.textContent =
            "✅ Dataset predictions completed successfully!";

        status.style.color = "#15803d";

        console.log(
            "Dataset prediction summary:",
            data
        );

        results.scrollIntoView({

            behavior: "smooth",

            block: "start"

        });

    } catch (error) {

        console.error(
            "Dataset prediction error:",
            error
        );

        status.textContent =
            "❌ Prediction failed: " + error.message;

        status.style.color = "#dc2626";

    } finally {

        button.disabled = false;

        button.textContent =
            originalText || "🚀 Run Prediction on Entire Dataset";
    }
}


// =====================================================
// 10. INITIALIZE APPLICATION
// =====================================================

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const predictionForm = document.getElementById(
            "prediction-form"
        );

        if (predictionForm) {

            predictionForm.addEventListener(
                "submit",
                handleIndividualPrediction
            );

        } else {

            console.warn(
                "Prediction form was not found."
            );
        }

        loadDashboard();

        loadModelInfo();

    }
);