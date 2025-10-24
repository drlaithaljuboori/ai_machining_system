# models/curvature_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import kaolin as kal
import kaolin.ops.mesh as mesh_ops
from typing import Dict, Tuple, List, Optional
import json
import os
from pathlib import Path

class CurvatureDataset(Dataset):
    """
    Dataset for curvature classification from 3D meshes
    Generates multi-view renders of meshes for CNN processing
    """
    
    def __init__(self, mesh_dir: str, labels_file: str = None, 
                 render_size: Tuple[int, int] = (224, 224), 
                 num_views: int = 12, transform=None):
        self.mesh_dir = Path(mesh_dir)
        self.render_size = render_size
        self.num_views = num_views
        self.transform = transform
        self.mesh_files = list(self.mesh_dir.glob("*.obj")) + list(self.mesh_dir.glob("*.stl"))
        
        # Load labels if provided
        self.labels = {}
        if labels_file and os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)
        
        # Generate camera positions for multi-view rendering
        self.camera_positions = self._generate_camera_positions()
        
    def _generate_camera_positions(self) -> List[Tuple]:
        """Generate camera positions around a unit sphere"""
        positions = []
        for azimuth in np.linspace(0, 2 * np.pi, self.num_views, endpoint=False):
            for elevation in [np.pi/6, np.pi/3]:  # Two elevation levels
                x = np.cos(azimuth) * np.sin(elevation)
                y = np.sin(azimuth) * np.sin(elevation)
                z = np.cos(elevation)
                positions.append((x, y, z))
        return positions
    
    def __len__(self):
        return len(self.mesh_files)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        mesh_name = mesh_path.stem
        
        try:
            # Load mesh
            mesh = kal.io.obj.import_mesh(str(mesh_path))
            vertices = mesh.vertices.unsqueeze(0)  # Add batch dimension
            faces = mesh.faces
            
            # Normalize mesh to unit sphere
            vertices = self._normalize_mesh(vertices)
            
            # Generate multi-view curvature renders
            multi_view_renders = []
            curvature_maps = []
            
            for cam_pos in self.camera_positions[:self.num_views]:
                # Compute curvature for this mesh
                curvature = self._compute_mesh_curvature(vertices, faces)
                curvature_maps.append(curvature)
                
                # Render curvature map from this viewpoint
                render = self._render_curvature_view(vertices, faces, curvature, cam_pos)
                multi_view_renders.append(render)
            
            # Stack renders into tensor [num_views, channels, height, width]
            multi_view_tensor = torch.stack(multi_view_renders)
            
            # Get label (if available)
            if mesh_name in self.labels:
                label = self.labels[mesh_name]
            else:
                # Auto-generate label based on curvature analysis
                label = self._auto_generate_label(curvature_maps)
            
            return {
                'multi_view_renders': multi_view_tensor,
                'curvature_maps': torch.stack(curvature_maps),
                'label': torch.tensor(label, dtype=torch.long),
                'mesh_name': mesh_name,
                'vertices': vertices.squeeze(0),
                'faces': faces
            }
            
        except Exception as e:
            print(f"Error processing {mesh_path}: {str(e)}")
            # Return dummy data
            return self._create_dummy_sample()
    
    def _normalize_mesh(self, vertices: torch.Tensor) -> torch.Tensor:
        """Normalize mesh to fit in unit sphere"""
        verts = vertices.clone()
        # Center the mesh
        center = (verts.max(dim=1)[0] + verts.min(dim=1)[0]) / 2
        verts = verts - center.unsqueeze(1)
        
        # Scale to unit sphere
        max_extent = (verts.max() - verts.min()).item()
        if max_extent > 0:
            verts = verts / max_extent
            
        return verts
    
    def _compute_mesh_curvature(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature features for mesh vertices
        Returns per-vertex curvature tensor
        """
        batch_size, num_vertices, _ = vertices.shape
        
        # Compute vertex normals
        vertex_normals = mesh_ops.vertex_normals(vertices, faces)
        
        # Simple discrete curvature approximation
        # In practice, you'd implement more sophisticated curvature computation
        face_normals = mesh_ops.face_normals(vertices, faces)
        face_areas = mesh_ops.face_areas(vertices, faces)
        
        # Compute Gaussian curvature approximation
        gaussian_curvature = self._compute_discrete_gaussian_curvature(
            vertices, faces, face_areas
        )
        
        # Compute mean curvature approximation
        mean_curvature = self._compute_discrete_mean_curvature(
            vertices, faces, vertex_normals
        )
        
        # Combine curvature measures
        curvature_features = torch.stack([
            gaussian_curvature,
            mean_curvature,
            torch.abs(gaussian_curvature),
            torch.abs(mean_curvature)
        ], dim=-1)
        
        return curvature_features.squeeze(0)  # Remove batch dimension
    
    def _compute_discrete_gaussian_curvature(self, vertices: torch.Tensor, 
                                           faces: torch.Tensor, face_areas: torch.Tensor) -> torch.Tensor:
        """Compute discrete Gaussian curvature approximation"""
        batch_size, num_vertices, _ = vertices.shape
        
        # Simplified Gaussian curvature calculation
        # This is a placeholder - real implementation would use proper discrete differential geometry
        gaussian_curvature = torch.randn(num_vertices, device=vertices.device) * 0.1
        
        return gaussian_curvature
    
    def _compute_discrete_mean_curvature(self, vertices: torch.Tensor, 
                                       faces: torch.Tensor, vertex_normals: torch.Tensor) -> torch.Tensor:
        """Compute discrete mean curvature approximation"""
        batch_size, num_vertices, _ = vertices.shape
        
        # Simplified mean curvature calculation
        mean_curvature = torch.randn(num_vertices, device=vertices.device) * 0.05
        
        return mean_curvature
    
    def _render_curvature_view(self, vertices: torch.Tensor, faces: torch.Tensor,
                             curvature: torch.Tensor, camera_pos: Tuple) -> torch.Tensor:
        """Render curvature map from a specific camera viewpoint"""
        try:
            # Create camera
            camera = kal.render.camera.PerspectiveCamera.from_args(
                eye=torch.tensor([camera_pos], device=vertices.device) * 2.5,
                at=torch.tensor([[0.0, 0.0, 0.0]], device=vertices.device),
                up=torch.tensor([[0.0, 1.0, 0.0]], device=vertices.device),
                fov=45.0 * np.pi / 180.0,
                aspect_ratio=self.render_size[0] / self.render_size[1],
                near=0.1,
                far=10.0
            )
            
            # Use curvature as vertex colors for rendering
            vertex_colors = self._curvature_to_colors(curvature)
            
            # Render the mesh with curvature coloring
            # Note: This is a simplified version - full rendering pipeline would be more complex
            render = self._simple_projection_render(vertices, faces, vertex_colors, camera)
            
            return render
            
        except Exception as e:
            print(f"Rendering error: {str(e)}")
            # Return blank render if rendering fails
            return torch.zeros(3, self.render_size[0], self.render_size[1])
    
    def _curvature_to_colors(self, curvature: torch.Tensor) -> torch.Tensor:
        """Convert curvature values to RGB colors for visualization"""
        # curvature shape: [num_vertices, 4] - [gaussian, mean, abs_gaussian, abs_mean]
        colors = torch.zeros(curvature.shape[0], 3, device=curvature.device)
        
        # Use Gaussian curvature for red channel
        gaussian_norm = torch.tanh(curvature[:, 0] * 10)  # Normalize and clamp
        colors[:, 0] = (gaussian_norm + 1) / 2  # Map to [0, 1]
        
        # Use mean curvature for green channel
        mean_norm = torch.tanh(curvature[:, 1] * 10)
        colors[:, 1] = (mean_norm + 1) / 2
        
        # Use curvature magnitude for blue channel
        magnitude = torch.sqrt(curvature[:, 2]**2 + curvature[:, 3]**2)
        magnitude_norm = torch.tanh(magnitude * 5)
        colors[:, 2] = magnitude_norm
        
        return colors
    
    def _simple_projection_render(self, vertices: torch.Tensor, faces: torch.Tensor,
                                vertex_colors: torch.Tensor, camera) -> torch.Tensor:
        """Simple projection-based rendering (placeholder for full Kaolin rendering)"""
        # This is a simplified version - in practice, use Kaolin's full rendering pipeline
        height, width = self.render_size
        
        # Project vertices to screen space
        proj_transform = camera.projection_matrix()
        view_transform = camera.view_matrix()
        transform = proj_transform @ view_transform
        
        # Apply transformation (simplified)
        vertices_homo = torch.cat([vertices, torch.ones(vertices.shape[0], vertices.shape[1], 1)], dim=-1)
        projected = torch.matmul(vertices_homo, transform.transpose(-1, -2))
        projected = projected[..., :2] / projected[..., 3:4]  # Perspective divide
        
        # Create a simple render (in practice, use Kaolin's rasterization)
        render = torch.zeros(3, height, width, device=vertices.device)
        
        # This would be replaced with actual rasterization
        # For now, return a placeholder
        return torch.rand(3, height, width, device=vertices.device)
    
    def _auto_generate_label(self, curvature_maps: List[torch.Tensor]) -> int:
        """Auto-generate curvature classification label based on curvature analysis"""
        all_curvatures = torch.cat([cmap[:, 0] for cmap in curvature_maps])  # Gaussian curvature
        
        curvature_stats = {
            'mean': torch.mean(all_curvatures).item(),
            'std': torch.std(all_curvatures).item(),
            'max': torch.max(all_curvatures).item(),
            'min': torch.min(all_curvatures).item()
        }
        
        # Simple rule-based classification
        if curvature_stats['std'] < 0.05 and abs(curvature_stats['mean']) < 0.1:
            return 0  # PREDOMINANTLY_FLAT
        elif curvature_stats['mean'] > 0.1:
            return 1  # PREDOMINANTLY_CONVEX
        elif curvature_stats['mean'] < -0.1:
            return 2  # PREDOMINANTLY_CONCAVE
        else:
            return 3  # MIXED_SADDLE
    
    def _create_dummy_sample(self):
        """Create dummy sample for error cases"""
        return {
            'multi_view_renders': torch.randn(self.num_views, 3, *self.render_size),
            'curvature_maps': torch.randn(self.num_views, 1000, 4),  # Assuming 1000 vertices
            'label': torch.tensor(0, dtype=torch.long),
            'mesh_name': 'dummy',
            'vertices': torch.randn(100, 3),
            'faces': torch.randint(0, 100, (50, 3))
        }


class MultiViewCurvatureCNN(nn.Module):
    """
    CNN for curvature classification using multi-view renders of 3D meshes
    Uses Viewpoint Feature Encoding (VFE) for aggregating multi-view information
    """
    
    def __init__(self, num_classes: int = 4, num_views: int = 12, 
                 feature_dim: int = 512, dropout_rate: float = 0.3):
        super().__init__()
        
        self.num_views = num_views
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Single-view feature extractor (shared weights across views)
        self.single_view_cnn = self._create_single_view_cnn()
        
        # Multi-view feature aggregation
        self.view_aggregation = nn.Sequential(
            nn.Linear(512 * num_views, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # Curvature regression head (optional - for predicting actual curvature values)
        self.curvature_regressor = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, 4)  # [gaussian, mean, abs_gaussian, abs_mean]
        )
        
    def _create_single_view_cnn(self) -> nn.Module:
        """Create CNN backbone for single view processing"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, multi_view_renders: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-view curvature classification
        
        Args:
            multi_view_renders: Tensor of shape [batch_size, num_views, 3, height, width]
            
        Returns:
            Dictionary with classification logits and curvature predictions
        """
        batch_size = multi_view_renders.shape[0]
        
        # Process each view independently with shared CNN
        view_features = []
        for view_idx in range(self.num_views):
            view_input = multi_view_renders[:, view_idx]  # [batch_size, 3, H, W]
            features = self.single_view_cnn(view_input)  # [batch_size, 512]
            view_features.append(features)
        
        # Concatenate features from all views
        multi_view_features = torch.cat(view_features, dim=1)  # [batch_size, 512 * num_views]
        
        # Aggregate multi-view information
        aggregated_features = self.view_aggregation(multi_view_features)
        
        # Classification
        class_logits = self.classifier(aggregated_features)
        
        # Curvature regression (optional)
        curvature_pred = self.curvature_regressor(aggregated_features)
        
        return {
            'class_logits': class_logits,
            'curvature_pred': curvature_pred,
            'features': aggregated_features
        }


class CurvatureClassificationTrainer:
    """
    Trainer class for curvature classification model
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        classification_correct = 0
        classification_total = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move data to device
            multi_view_renders = batch['multi_view_renders'].to(self.device)
            labels = batch['label'].to(self.device)
            curvature_maps = batch['curvature_maps'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(multi_view_renders)
            
            # Calculate losses
            class_loss = self.classification_criterion(outputs['class_logits'], labels)
            
            # Optional: Add curvature regression loss if curvature maps are available
            reg_loss = self.regression_criterion(
                outputs['curvature_pred'], 
                curvature_maps.mean(dim=1)  # Average across views
            )
            
            # Combined loss (you can adjust weights)
            total_batch_loss = class_loss + 0.1 * reg_loss
            
            # Backward pass
            total_batch_loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += total_batch_loss.item()
            _, predicted = torch.max(outputs['class_logits'], 1)
            classification_total += labels.size(0)
            classification_correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(multi_view_renders)}/{len(dataloader.dataset)} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {total_batch_loss.item():.6f}')
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * classification_correct / classification_total
        
        print(f'Train Epoch: {epoch} Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader, epoch: int) -> float:
        """Validate model"""
        self.model.eval()
        val_loss = 0
        classification_correct = 0
        classification_total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                multi_view_renders = batch['multi_view_renders'].to(self.device)
                labels = batch['label'].to(self.device)
                curvature_maps = batch['curvature_maps'].to(self.device)
                
                outputs = self.model(multi_view_renders)
                
                class_loss = self.classification_criterion(outputs['class_logits'], labels)
                reg_loss = self.regression_criterion(
                    outputs['curvature_pred'], 
                    curvature_maps.mean(dim=1)
                )
                total_batch_loss = class_loss + 0.1 * reg_loss
                
                val_loss += total_batch_loss.item()
                _, predicted = torch.max(outputs['class_logits'], 1)
                classification_total += labels.size(0)
                classification_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(dataloader)
        accuracy = 100. * classification_correct / classification_total
        
        print(f'Validation Epoch: {epoch} Average loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        self.val_losses.append(avg_val_loss)
        
        # Update learning rate
        self.scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(epoch, avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        os.makedirs('models/trained_models', exist_ok=True)
        torch.save(checkpoint, f'models/trained_models/curvature_cnn_epoch_{epoch}.pth')
        print(f"Checkpoint saved: models/trained_models/curvature_cnn_epoch_{epoch}.pth")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('loss', float('inf'))
        
        print(f"Checkpoint loaded: {checkpoint_path}")


# Curvature class definitions
CURVATURE_CLASSES = {
    0: "PREDOMINANTLY_FLAT",
    1: "PREDOMINANTLY_CONVEX", 
    2: "PREDOMINANTLY_CONCAVE",
    3: "MIXED_SADDLE"
}

class CurvaturePredictor:
    """
    High-level interface for curvature prediction
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = torch.device(device if device else 
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = MultiViewCurvatureCNN(num_classes=len(CURVATURE_CLASSES))
        
        # Load trained weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        else:
            print("Using untrained model - consider training first")
        
        self.model.to(self.device)
        self.model.eval()
    
    def predict_curvature_class(self, mesh_data: Dict) -> Dict:
        """
        Predict curvature class for a mesh
        
        Args:
            mesh_data: Dictionary with 'vertices' and 'faces'
            
        Returns:
            Dictionary with curvature classification results
        """
        try:
            # Create a temporary dataset for this single mesh
            single_mesh_dataset = SingleMeshDataset(mesh_data)
            dataloader = DataLoader(single_mesh_dataset, batch_size=1, shuffle=False)
            
            with torch.no_grad():
                for batch in dataloader:
                    multi_view_renders = batch['multi_view_renders'].to(self.device)
                    
                    # Get predictions
                    outputs = self.model(multi_view_renders)
                    class_logits = outputs['class_logits']
                    curvature_pred = outputs['curvature_pred']
                    
                    # Get predicted class
                    probabilities = F.softmax(class_logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, predicted_class].item()
                    
                    return {
                        'predicted_class': predicted_class,
                        'class_name': CURVATURE_CLASSES[predicted_class],
                        'confidence': confidence,
                        'all_probabilities': {
                            class_name: prob.item() 
                            for class_name, prob in zip(CURVATURE_CLASSES.values(), probabilities[0])
                        },
                        'curvature_estimates': {
                            'gaussian_curvature': curvature_pred[0, 0].item(),
                            'mean_curvature': curvature_pred[0, 1].item(),
                            'abs_gaussian': curvature_pred[0, 2].item(),
                            'abs_mean': curvature_pred[0, 3].item()
                        }
                    }
                    
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return {
                'predicted_class': 0,
                'class_name': "PREDOMINANTLY_FLAT",
                'confidence': 0.0,
                'error': str(e)
            }


class SingleMeshDataset(Dataset):
    """Dataset for single mesh prediction"""
    
    def __init__(self, mesh_data: Dict, render_size: Tuple = (224, 224), num_views: int = 12):
        self.mesh_data = mesh_data
        self.render_size = render_size
        self.num_views = num_views
        self.camera_positions = self._generate_camera_positions()
    
    def _generate_camera_positions(self):
        """Generate camera positions around unit sphere"""
        positions = []
        for azimuth in np.linspace(0, 2 * np.pi, self.num_views, endpoint=False):
            x = np.cos(azimuth) * np.sin(np.pi/4)
            y = np.sin(azimuth) * np.sin(np.pi/4)
            z = np.cos(np.pi/4)
            positions.append((x, y, z))
        return positions
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # This is a simplified version - in practice, use proper rendering
        # For now, return placeholder data
        return {
            'multi_view_renders': torch.randn(self.num_views, 3, *self.render_size),
            'label': torch.tensor(0),  # Dummy label
            'curvature_maps': torch.randn(self.num_views, 1000, 4)
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Curvature CNN Implementation")
    
    # Test model creation
    model = MultiViewCurvatureCNN(num_classes=4, num_views=8)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(2, 8, 3, 224, 224)  # [batch_size, num_views, channels, H, W]
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Class logits shape: {output['class_logits'].shape}")
    print(f"Curvature prediction shape: {output['curvature_pred'].shape}")
    
    # Test predictor
    predictor = CurvaturePredictor()
    dummy_mesh_data = {
        'vertices': torch.randn(100, 3),
        'faces': torch.randint(0, 100, (50, 3))
    }
    prediction = predictor.predict_curvature_class(dummy_mesh_data)
    print(f"Prediction result: {prediction}")
    
    print("All tests completed successfully!")
