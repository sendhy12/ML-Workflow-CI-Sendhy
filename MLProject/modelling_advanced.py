import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
import joblib
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import threading
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_mlflow():
    """Setup MLflow experiment for advanced level"""
    experiment_name = "CI_ML_Experiment_Advanced"
    
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ Experiment '{experiment_name}' set successfully")
    except Exception as e:
        logger.warning(f"⚠️  Creating new experiment: {experiment_name}")
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            logger.info("✅ Experiment created and set")
        except Exception as e2:
            logger.error(f"❌ Failed to create experiment: {e2}")
            raise

def load_and_validate_data():
    """Load and validate data with comprehensive checks"""
    try:
        # Load data
        X_train = pd.read_csv('./dataset_preprocessing/X_train.csv')
        X_test = pd.read_csv('./dataset_preprocessing/X_test.csv')
        y_train = pd.read_csv('./dataset_preprocessing/y_train.csv')['target'].values
        y_test = pd.read_csv('./dataset_preprocessing/y_test.csv')['target'].values

        # Comprehensive validation
        assert len(X_train) == len(y_train), "Training data length mismatch"
        assert len(X_test) == len(y_test), "Test data length mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"
        assert not X_train.isnull().any().any(), "Training data contains null values"
        assert not X_test.isnull().any().any(), "Test data contains null values"
        assert set(np.unique(y_train)) == set(np.unique(y_test)), "Label mismatch between train/test"

        logger.info(f"📊 Data loaded and validated successfully:")
        logger.info(f"   - Training: {X_train.shape}, Labels: {len(y_train)}")
        logger.info(f"   - Testing: {X_test.shape}, Labels: {len(y_test)}")
        logger.info(f"   - Features: {list(X_train.columns)}")
        logger.info(f"   - Classes: {np.unique(y_train)}")

        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"❌ Error loading data: {e}")
        raise

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning with GridSearchCV"""
    logger.info("🔍 Starting hyperparameter tuning...")
    
    # Define parameter grid (simplified for CI efficiency)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='accuracy', 
        n_jobs=1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    logger.info(f"✅ Best parameters: {grid_search.best_params_}")
    logger.info(f"✅ Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

def create_advanced_artifacts(model, X_train, X_test, y_test, y_pred, y_pred_proba, run_id):
    """Create comprehensive artifacts for advanced level"""
    
    artifacts = {}
    
    # 1. Enhanced Confusion Matrix with detailed annotations
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Introvert', 'Extrovert'],
               yticklabels=['Introvert', 'Extrovert'],
               cbar_kws={'label': 'Count'})
    
    # Add percentages and additional info
    total = np.sum(cm)
    accuracy = np.trace(cm) / total
    
    for i in range(2):
        for j in range(2):
            plt.text(j+0.5, i+0.8, f'{cm[i,j]/total:.1%}',
                    ha='center', va='center', fontsize=12, 
                    color='red', fontweight='bold')
    
    plt.title(f'Advanced CI - Detailed Confusion Matrix\n'
              f'Run: {run_id} | Accuracy: {accuracy:.3f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = f'confusion_matrix_advanced_{run_id}.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['confusion_matrix'] = cm_path

    # 2. ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - Advanced CI\nRun: {run_id}', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = f'roc_curve_advanced_{run_id}.png'
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['roc_curve'] = roc_path

    # 3. Feature Importance with detailed analysis
    feature_names = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(14, 10))
    
    # Create horizontal bar plot for better readability
    y_pos = np.arange(len(feature_names))
    bars = plt.barh(y_pos, importances[indices], 
                    color='skyblue', alpha=0.8, edgecolor='navy')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', fontweight='bold')
    
    plt.yticks(y_pos, [feature_names[i] for i in indices])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Feature Importance Analysis - Advanced CI\n'
              f'Run: {run_id} | Total Features: {len(feature_names)}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    fi_path = f'feature_importance_advanced_{run_id}.png'
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['feature_importance'] = fi_path

    # 4. Comprehensive metrics dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Metrics comparison
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    bars1 = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    for bar, value in zip(bars1, values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_title('Performance Metrics', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Class distribution
    unique, counts = np.unique(y_test, return_counts=True)
    ax2.pie(counts, labels=['Introvert', 'Extrovert'], autopct='%1.1f%%',
            colors=['lightcoral', 'lightblue'], startangle=90)
    ax2.set_title('Test Set Class Distribution', fontweight='bold')
    
    # Prediction confidence distribution
    confidence = np.max(y_pred_proba, axis=1)
    ax3.hist(confidence, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax3.axvline(confidence.mean(), color='red', linestyle='--', 
                label=f'Mean: {confidence.mean():.3f}')
    ax3.set_xlabel('Prediction Confidence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Feature importance top 10
    top_10_idx = indices[:10]
    ax4.barh(range(10), importances[top_10_idx], color='purple', alpha=0.7)
    ax4.set_yticks(range(10))
    ax4.set_yticklabels([feature_names[i] for i in top_10_idx])
    ax4.set_xlabel('Importance Score')
    ax4.set_title('Top 10 Most Important Features', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Advanced ML Dashboard - Run: {run_id}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    dashboard_path = f'ml_dashboard_advanced_{run_id}.png'
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts['ml_dashboard'] = dashboard_path

    return artifacts

def create_model_api():
    """Create Flask API for model serving"""
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            # This is a placeholder - in real deployment, load the trained model
            data = request.json
            return jsonify({
                "status": "success",
                "prediction": "placeholder",
                "confidence": 0.85,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400
    
    return app

def save_model_artifacts(model, model_metadata, run_id):
    """Save model and metadata for deployment"""
    
    # Save model
    model_path = f'models/model_advanced_{run_id}.pkl'
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata_path = f'models/metadata_advanced_{run_id}.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    logger.info(f"✅ Model saved: {model_path}")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    
    return model_path, metadata_path

def train_advanced_model():
    """Train model with advanced features: hyperparameter tuning, comprehensive logging"""
    
    logger.info("🎯 Starting CI ML Training - Advanced Level")
    logger.info("=" * 70)
    
    # Setup MLflow
    setup_mlflow()
    
    # Generate unique run ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"CI_Advanced_Training_{run_id}"
    
    with mlflow.start_run(run_name=run_name):
        try:
            # Load and validate data
            X_train, X_test, y_train, y_test = load_and_validate_data()

            # Hyperparameter tuning
            logger.info("🔧 Performing hyperparameter optimization...")
            best_model, best_params, best_cv_score = hyperparameter_tuning(X_train, y_train)

            # Train final model with best parameters
            logger.info("🤖 Training final model with optimized parameters...")
            final_model = RandomForestClassifier(**best_params, random_state=42, n_jobs=1)
            final_model.fit(X_train, y_train)

            # Comprehensive evaluation
            logger.info("📊 Performing comprehensive evaluation...")
            
            # Cross-validation
            cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            # Predictions
            y_pred = final_model.predict(X_test)
            y_pred_proba = final_model.predict_proba(X_test)

            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])

            # Log parameters
            mlflow.log_params(best_params)
            mlflow.log_param("hyperparameter_tuning", True)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("run_id", run_id)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])

            # Log comprehensive metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("auc_score", auc_score)
            mlflow.log_metric("cv_mean_accuracy", cv_mean)
            mlflow.log_metric("cv_std_accuracy", cv_std)
            mlflow.log_metric("best_cv_score", best_cv_score)

            # Log model with signature
            mlflow.sklearn.log_model(
                final_model, 
                "model", 
                registered_model_name="PersonalityClassifier_CI_Advanced"
            )

            # Create comprehensive artifacts
            logger.info("🎨 Creating comprehensive artifacts...")
            artifacts = create_advanced_artifacts(
                final_model, X_train, X_test, y_test, y_pred, y_pred_proba, run_id
            )

            # Log all artifacts
            for artifact_name, artifact_path in artifacts.items():
                mlflow.log_artifact(artifact_path)
                logger.info(f"   ✅ Logged: {artifact_name}")

            # Create detailed metadata
            model_metadata = {
                "experiment_info": {
                    "run_id": run_id,
                    "run_name": run_name,
                    "timestamp": datetime.now().isoformat(),
                    "mlflow_version": mlflow.__version__
                },
                "model_config": {
                    "best_params": best_params,
                    "hyperparameter_tuning": True,
                    "cv_folds": 5
                },
                "data_info": {
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "n_features": X_train.shape[1],
                    "feature_names": X_train.columns.tolist()
                },
                "performance_metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "auc_score": auc_score
                },
                "cross_validation": {
                    "cv_scores": cv_scores.tolist(),
                    "cv_mean": cv_mean,
                    "cv_std": cv_std,
                    "best_cv_score": best_cv_score
                },
                "feature_importance": dict(zip(X_train.columns, final_model.feature_importances_)),
                "deployment_ready": True
            }

            # Save model artifacts for deployment
            model_path, metadata_path = save_model_artifacts(final_model, model_metadata, run_id)
            
            # Log deployment artifacts
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(metadata_path)

            # Create and log comprehensive results
            results_path = f'advanced_results_{run_id}.json'
            with open(results_path, 'w') as f:
                json.dump(model_metadata, f, indent=2)
            
            mlflow.log_artifact(results_path)

            # Print comprehensive results
            logger.info(f"\n=== ADVANCED CI TRAINING RESULTS ===")
            logger.info(f"🏆 Test Accuracy: {accuracy:.4f}")
            logger.info(f"🎯 Precision: {precision:.4f}")
            logger.info(f"📊 Recall: {recall:.4f}")
            logger.info(f"🔄 F1-Score: {f1:.4f}")
            logger.info(f"📈 AUC Score: {auc_score:.4f}")
            logger.info(f"✅ CV Score: {cv_mean:.4f} ± {cv_std:.4f}")
            logger.info(f"🔧 Best CV Score: {best_cv_score:.4f}")

            logger.info(f"\n=== DEPLOYMENT ARTIFACTS ===")
            logger.info(f"🤖 Model: {model_path}")
            logger.info(f"📄 Metadata: {metadata_path}")
            for artifact_name, artifact_path in artifacts.items():
                logger.info(f"📊 {artifact_name}: {artifact_path}")

            logger.info(f"\n=== CLASSIFICATION REPORT ===")
            print(classification_report(y_test, y_pred, target_names=['Introvert', 'Extrovert']))

            # Cleanup local files (artifacts are logged to MLflow)
            for artifact_path in artifacts.values():
                if os.path.exists(artifact_path):
                    os.remove(artifact_path)
            
            for file_path in [results_path, model_path, metadata_path]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            return final_model, accuracy, model_metadata

        except Exception as e:
            logger.error(f"❌ Error during advanced training: {e}")
            mlflow.log_param("error", str(e))
            raise

def run_api_server():
    """Run Flask API server in background for testing"""
    app = create_model_api()
    # Safer for local testing
    app.run(host='127.0.0.1', port=5000, debug=False)


if __name__ == "__main__":
    try:
        logger.info("🚀 Starting Advanced CI ML Training with Deployment...")
        
        # Train model
        model, accuracy, metadata = train_advanced_model()
        
        logger.info(f"\n🎉 Advanced training completed successfully!")
        logger.info(f"🏆 Final accuracy: {accuracy:.4f}")
        logger.info(f"🚀 Model is deployment-ready!")
        
        # Start API server in background for testing (optional in CI)
        if os.getenv('START_API', 'false').lower() == 'true':
            logger.info("🌐 Starting API server for testing...")
            api_thread = threading.Thread(target=run_api_server, daemon=True)
            api_thread.start()
            time.sleep(2)  # Give server time to start
            logger.info("✅ API server started on port 5000")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"\n💥 Advanced training failed: {e}")
        # Exit with error code for CI
        sys.exit(1)