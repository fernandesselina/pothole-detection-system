import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import torch
from pathlib import Path
import streamlit as st
from PIL import Image
import yaml
from ultralytics import YOLO

class PotholeDatasetManager:
    """Handles dataset download, preprocessing, and YOLO format conversion"""
    
    def __init__(self, data_dir="pothole_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.images_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
        
    def download_rdd2022_sample(self):
        """Download a sample dataset for pothole detection"""
        print("Setting up sample dataset structure...")
        
        # Create train/val/test directories
        for split in ['train', 'val', 'test']:
            (self.images_dir / split).mkdir(exist_ok=True)
            (self.labels_dir / split).mkdir(exist_ok=True)
        
        # Create dataset.yaml for YOLO
        dataset_config = {
            'path': str(self.data_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'names': {0: 'pothole'},
            'nc': 1
        }
        
        with open(self.data_dir / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f)
            
        print(f"Dataset structure created at {self.data_dir}")
        return str(self.data_dir / 'dataset.yaml')
    
    def create_sample_data(self):
        """Create sample synthetic data for demonstration"""
        print("Creating sample synthetic pothole data...")
        
        # Create some sample annotations
        sample_annotations = [
            # Format: class_id x_center y_center width height (normalized)
            "0 0.5 0.6 0.3 0.2",  # pothole in center-bottom
            "0 0.3 0.4 0.2 0.15", # smaller pothole left
            "0 0.7 0.7 0.25 0.18" # pothole bottom-right
        ]
        
        # Create sample data for each split
        splits = {'train': 50, 'val': 10, 'test': 10}
        
        for split, count in splits.items():
            for i in range(count):
                # Create dummy image (you'll replace with real RDD2022 data)
                img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                img_path = self.images_dir / split / f"sample_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)
                
                # Create corresponding label
                label_path = self.labels_dir / split / f"sample_{i:03d}.txt"
                with open(label_path, 'w') as f:
                    # Randomly select 1-2 annotations
                    num_objects = np.random.randint(1, 3)
                    selected_annotations = np.random.choice(sample_annotations, num_objects, replace=False)
                    f.write('\n'.join(selected_annotations))
        
        print(f"Sample dataset created with {sum(splits.values())} images")

class PotholeDetector:
    """YOLO-based pothole detection system"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.write(f"Using device: {self.device}")
        
    def train_model(self, dataset_yaml, epochs=50, img_size=640):
        """Train YOLO model on pothole dataset"""
        st.write("Training started...")
        
        # Initialize YOLO model
        self.model = YOLO('yolov8n.pt') 
        
        # Train the model
        results = self.model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=16,
            device=self.device,
            project='pothole_runs',
            name='pothole_detection',
            save=True,
            plots=True
        )
        
        # Save the best model
        self.model_path = results.save_dir / 'weights' / 'best.pt'
        st.success("Model training completed")
        return self.model_path
    
    def load_model(self, model_path):
        """Load pre-trained YOLO model"""
        self.model_path = model_path
        self.model = YOLO(model_path)
        st.success("Model loaded successfully")
    
    def detect_potholes(self, image_path, confidence_threshold=0.5):
        """Detect potholes in an image"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Run inference
        results = self.model(image_path, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'class': 'pothole',
                        'class_id': int(class_id)
                    }
                    detections.append(detection)
        
        return detections, results[0].plot() if results else None
    
    def evaluate_model(self, dataset_yaml):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Run validation
        metrics = self.model.val(data=dataset_yaml)
        
        return {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': metrics.box.mp,
            'recall': metrics.box.mr
        }

class ReportManager:
    """Handle pothole report submission and storage"""
    
    def __init__(self, reports_file="pothole_reports.json"):
        self.reports_file = reports_file
        self.reports = self.load_reports()
    
    def load_reports(self):
        """Load existing reports from file"""
        try:
            with open(self.reports_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_reports(self):
        """Save reports to file"""
        with open(self.reports_file, 'w') as f:
            json.dump(self.reports, f, indent=2, default=str)
    
    def create_report(self, image_path, detections, gps_coords=None, user_id="anonymous"):
        """Create a pothole report"""
        report = {
            'report_id': len(self.reports) + 1,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'image_path': str(image_path),
            'gps_coordinates': gps_coords or {'lat': None, 'lon': None},
            'detections': detections,
            'num_potholes': len(detections),
            'status': 'pending',
            'severity': self.classify_severity(detections)
        }
        
        self.reports.append(report)
        self.save_reports()
        return report
    
    def classify_severity(self, detections):
        """Classify pothole severity based on detection confidence and size"""
        if not detections:
            return 'none'
        
        max_confidence = max(det['confidence'] for det in detections)
        avg_size = np.mean([
            (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
            for det in detections
        ])
        
        if max_confidence > 0.8 and avg_size > 10000:
            return 'major'
        elif max_confidence > 0.6 and avg_size > 5000:
            return 'moderate'
        else:
            return 'minor'
    
    def get_reports_summary(self):
        """Get summary of all reports"""
        if not self.reports:
            return {}
        
        df = pd.DataFrame(self.reports)
        summary = {
            'total_reports': len(self.reports),
            'pending_reports': len(df[df['status'] == 'pending']),
            'resolved_reports': len(df[df['status'] == 'resolved']),
            'severity_distribution': df['severity'].value_counts().to_dict(),
            'avg_potholes_per_report': df['num_potholes'].mean()
        }
        return summary

class PotholeDetectionApp:
    """Streamlit web interface for pothole detection"""
    
    def __init__(self):
        self.detector = PotholeDetector()
        self.report_manager = ReportManager()
        self.dataset_manager = PotholeDatasetManager()
    
    def run(self):
        """Run the Streamlit app"""
        st.set_page_config(
            page_title="Smart Pothole Detection System",
            page_icon="🛣️",
            layout="wide"
        )
        
        st.title("🛣️ Smart Pothole Detection & Reporting System")
        st.markdown("**Modules 1 & 2**: Image Acquisition, Detection, and Report Submission")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Choose a page:",
            ["Detection Interface", "Model Training", "Reports Dashboard", "Dataset Management"]
        )
        
        if page == "Detection Interface":
            self.detection_interface()
        elif page == "Model Training":
            self.training_interface()
        elif page == "Reports Dashboard":
            self.reports_dashboard()
        elif page == "Dataset Management":
            self.dataset_management()
    
    def detection_interface(self):
        """Main detection interface"""
        st.header("Pothole Detection Interface")
        
        # Model loading section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Image or Video")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'mp4', 'avi'],
                help="Upload an image or video of a road surface"
            )
        
        with col2:
            st.subheader("Model Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1
            )
            
            # GPS coordinates (optional)
            st.subheader("GPS Coordinates (Optional)")
            lat = st.number_input("Latitude", value=0.0, format="%.6f")
            lon = st.number_input("Longitude", value=0.0, format="%.6f")
        
        # Model loading
        model_path = st.text_input(
            "Model Path (optional)",
            help="Path to trained YOLO model. Leave empty to use default."
        )
        
        if self.detector.model is None:
            try:
                if model_path and os.path.exists(model_path):
                    self.detector.load_model(model_path)
                    st.success(f"Model loaded from {model_path}")
                else:
                    default_model_path = "runs/detect/train4/weights/best.pt"
                    self.detector.load_model(default_model_path)
                    st.info("Using trained YOLOv8 model by default")
            except Exception as e:
                st.error(f"Error loading model: {e}")

        
        # Detection section
        if uploaded_file is not None:
            # Save uploaded file temporarily
            file_extension = uploaded_file.name.split('.')[-1].lower()
            temp_path = f"temp_upload.{file_extension}"
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Display original image/video
            if file_extension in ['jpg', 'jpeg', 'png']:
                st.subheader("Original Image")
                st.image(temp_path, caption="Uploaded Image", use_column_width=True)
                
                if st.button("Detect Potholes") and self.detector.model is not None:
                    with st.spinner("Detecting potholes..."):
                        try:
                            detections, annotated_img = self.detector.detect_potholes(
                                temp_path, confidence_threshold
                            )
                            
                            # Display results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Detection Results")
                                if annotated_img is not None:
                                    st.image(annotated_img, caption="Detected Potholes")
                                else:
                                    st.info("No potholes detected")
                            
                            with col2:
                                st.subheader("Detection Details")
                                if detections:
                                    for i, det in enumerate(detections):
                                        st.write(f"**Pothole {i+1}:**")
                                        st.write(f"- Confidence: {det['confidence']:.2f}")
                                        st.write(f"- Location: ({det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}) to ({det['bbox'][2]:.0f}, {det['bbox'][3]:.0f})")
                                    
                                    # Create report
                                    if st.button("Submit Report"):
                                        gps_coords = {'lat': lat, 'lon': lon} if lat != 0.0 and lon != 0.0 else None
                                        report = self.report_manager.create_report(
                                            temp_path, detections, gps_coords
                                        )
                                        st.success(f"Report #{report['report_id']} submitted successfully!")
                                        st.json(report)
                                else:
                                    st.info("No potholes detected in the image")
                        
                        except Exception as e:
                            st.error(f"Error during detection: {e}")
            
            elif file_extension in ['mp4', 'avi']:
                st.info("Video processing feature coming soon!")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def training_interface(self):
        """Model training interface"""
        st.header("Model Training Interface")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Preparation")
            if st.button("Setup Sample RDD2022 Dataset"):
                with st.spinner("Setting up dataset..."):
                    dataset_yaml = self.dataset_manager.download_rdd2022_sample()
                    self.dataset_manager.create_sample_data()
                    st.success(f"Dataset prepared at: {dataset_yaml}")
                    st.session_state['dataset_yaml'] = dataset_yaml
        
        with col2:
            st.subheader("Training Parameters")
            epochs = st.number_input("Epochs", min_value=1, max_value=200, value=50)
            img_size = st.selectbox("Image Size", [416, 640, 832], index=1)
        
        # Training section
        if st.button("Start Training") and 'dataset_yaml' in st.session_state:
            with st.spinner("Training model... This may take a while."):
                try:
                    model_path = self.detector.train_model(
                        st.session_state['dataset_yaml'],
                        epochs=epochs,
                        img_size=img_size
                    )
                    st.success(f"Training completed! Model saved at: {model_path}")
                    st.session_state['trained_model_path'] = model_path
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        # Model evaluation
        if 'trained_model_path' in st.session_state:
            st.subheader("Model Evaluation")
            if st.button("Evaluate Model"):
                try:
                    metrics = self.detector.evaluate_model(st.session_state['dataset_yaml'])
                    st.success("Model evaluation completed!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("mAP@50", f"{metrics['mAP50']:.3f}")
                    with col2:
                        st.metric("mAP@50-95", f"{metrics['mAP50-95']:.3f}")
                    with col3:
                        st.metric("Precision", f"{metrics['precision']:.3f}")
                    with col4:
                        st.metric("Recall", f"{metrics['recall']:.3f}")
                
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
    
    def reports_dashboard(self):
        """Reports dashboard"""
        st.header("Pothole Reports Dashboard")
        
        # Summary statistics
        summary = self.report_manager.get_reports_summary()
        
        if summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Reports", summary['total_reports'])
            with col2:
                st.metric("Pending", summary['pending_reports'])
            with col3:
                st.metric("Resolved", summary['resolved_reports'])
            with col4:
                st.metric("Avg Potholes/Report", f"{summary['avg_potholes_per_report']:.1f}")
            
            # Severity distribution
            st.subheader("Severity Distribution")
            severity_data = summary['severity_distribution']
            fig, ax = plt.subplots()
            ax.pie(severity_data.values(), labels=severity_data.keys(), autopct='%1.1f%%')
            st.pyplot(fig)
            
            # Recent reports
            st.subheader("Recent Reports")
            if self.report_manager.reports:
                df = pd.DataFrame(self.report_manager.reports)
                df = df.sort_values('timestamp', ascending=False).head(10)
                st.dataframe(df[['report_id', 'timestamp', 'num_potholes', 'severity', 'status']])
        else:
            st.info("No reports found. Submit some detections first!")
    
    def dataset_management(self):
        """Dataset management interface"""
        st.header("Dataset Management")
        
        st.subheader("RDD2022 Dataset Information")
        st.markdown("""
        **Road Damage Dataset 2022 (RDD2022)** contains images of various road damages including potholes.
        
        For this demo, we'll use a sample dataset structure. In a real implementation, you would:
        1. Download the full RDD2022 dataset
        2. Filter for pothole-specific annotations
        3. Convert annotations to YOLO format
        4. Apply data augmentation using tools like Roboflow
        """)
        
        # Dataset statistics
        data_dir = Path("pothole_data")
        if data_dir.exists():
            st.subheader("Current Dataset Statistics")
            
            for split in ['train', 'val', 'test']:
                images_dir = data_dir / "images" / split
                if images_dir.exists():
                    num_images = len(list(images_dir.glob("*.jpg")))
                    st.write(f"**{split.capitalize()}**: {num_images} images")
        
        # Data augmentation options
        st.subheader("Data Augmentation Options")
        st.markdown("""
        Recommended augmentations for pothole detection:
        - **Rotation**: ±15 degrees
        - **Brightness**: ±20%
        - **Blur**: Gaussian blur (0-2 pixels)
        - **Horizontal Flip**: 50% probability
        - **Scaling**: 0.8-1.2x
        """)

def main():
    app = PotholeDetectionApp()
    app.run()

if __name__ == "__main__":
    main()
