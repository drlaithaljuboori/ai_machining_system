# core/geometry_processor.py
import torch
import kaolin as kal
import kaolin.ops.mesh as mesh_ops
import trimesh
import numpy as np
from typing import Dict, Tuple

class GeometryProcessor:
    def __init__(self, device='cuda'):
        self.device = device
        
    def process_cad_file(self, file_path: str) -> Tuple[Dict, Dict]:
        """
        Process CAD file and extract geometric features
        """
        try:
            # Load mesh using trimesh
            mesh = trimesh.load_mesh(file_path)
            
            # Convert to Kaolin format
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
            faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
            
            # Create Kaolin mesh object
            kal_mesh = kal.rep.TriangleMesh(vertices, faces)
            
            # Extract basic features
            features = self._extract_geometric_features(kal_mesh, mesh)
            
            mesh_data = {
                'vertices': vertices,
                'faces': faces,
                'kal_mesh': kal_mesh,
                'original_mesh': mesh
            }
            
            return mesh_data, features
            
        except Exception as e:
            raise Exception(f"Failed to process CAD file {file_path}: {str(e)}")
    
    def _extract_geometric_features(self, kal_mesh, original_mesh) -> Dict:
        """Extract key geometric features for analysis"""
        
        vertices = kal_mesh.vertices
        faces = kal_mesh.faces
        
        # Compute mesh properties
        face_normals = mesh_ops.face_normals(vertices.unsqueeze(0), faces)
        face_areas = mesh_ops.face_areas(vertices.unsqueeze(0), faces)
        
        # Bounding box dimensions
        bbox_min = torch.min(vertices, dim=0)[0]
        bbox_max = torch.max(vertices, dim=0)[0]
        bbox_dims = bbox_max - bbox_min
        
        # Volume and surface area
        volume = original_mesh.volume
        surface_area = original_mesh.area
        
        # Detect basic feature types
        features = {
            'bounding_box': {
                'min': bbox_min.cpu().numpy(),
                'max': bbox_max.cpu().numpy(),
                'dimensions': bbox_dims.cpu().numpy()
            },
            'basic_properties': {
                'volume': volume,
                'surface_area': surface_area,
                'face_count': len(faces),
                'vertex_count': len(vertices)
            },
            'detected_features': self._detect_manufacturing_features(original_mesh)
        }
        
        return features
    
    def _detect_manufacturing_features(self, mesh) -> Dict:
        """Detect common manufacturing features"""
        features = {
            'pockets': [],
            'holes': [],
            'fillets': [],
            'chamfers': [],
            'flat_faces': []
        }
        
        # Simple feature detection based on geometry
        # In practice, this would use more sophisticated algorithms
        
        # Detect flat faces (normals consistent)
        if hasattr(mesh, 'face_normals'):
            face_normals = mesh.face_normals
            # Simple flat region detection
            # This is a simplified version - real implementation would be more complex
            
        return features
