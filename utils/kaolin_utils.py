# utils/kaolin_utils.py
import torch
import kaolin as kal
import kaolin.ops.mesh as mesh_ops
import numpy as np
from typing import Tuple, Optional, Dict, Any

class KaolinHelper:
    """
    Helper class for Kaolin operations and utilities
    """
    
    @staticmethod
    def check_kaolin_installation() -> Dict[str, bool]:
        """
        Check if Kaolin is properly installed and accessible
        """
        checks = {
            'kaolin_imported': False,
            'cuda_available': False,
            'kaolin_ops_working': False,
            'version': None
        }
        
        try:
            import kaolin
            checks['kaolin_imported'] = True
            checks['version'] = kal.__version__
            
            # Check CUDA
            checks['cuda_available'] = torch.cuda.is_available()
            
            # Test basic operations
            if checks['cuda_available']:
                device = 'cuda'
                # Test mesh operations
                vertices = torch.randn(4, 3, device=device)
                faces = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)
                face_normals = mesh_ops.face_normals(vertices.unsqueeze(0), faces)
                checks['kaolin_ops_working'] = face_normals is not None
            else:
                # Test on CPU
                device = 'cpu'
                vertices = torch.randn(4, 3, device=device)
                faces = torch.tensor([[0, 1, 2], [1, 2, 3]], device=device)
                face_normals = mesh_ops.face_normals(vertices.unsqueeze(0), faces)
                checks['kaolin_ops_working'] = face_normals is not None
                
        except ImportError as e:
            print(f"Kaolin import error: {e}")
        except Exception as e:
            print(f"Kaolin operation error: {e}")
            
        return checks
    
    @staticmethod
    def normalize_mesh(vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize mesh to unit sphere
        """
        # Center the mesh
        verts_centered = vertices - vertices.mean(dim=0)
        
        # Scale to unit sphere
        max_extent = torch.max(verts_centered.abs())
        if max_extent > 0:
            verts_normalized = verts_centered / max_extent
        else:
            verts_normalized = verts_centered
            
        return verts_normalized, faces
    
    @staticmethod
    def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute face normals using Kaolin
        """
        return mesh_ops.face_normals(vertices.unsqueeze(0), faces).squeeze(0)
    
    @staticmethod
    def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute vertex normals using Kaolin
        """
        return mesh_ops.vertex_normals(vertices.unsqueeze(0), faces).squeeze(0)
    
    @staticmethod
    def compute_face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Compute face areas using Kaolin
        """
        return mesh_ops.face_areas(vertices.unsqueeze(0), faces).squeeze(0)
    
    @staticmethod
    def sample_points_on_mesh(vertices: torch.Tensor, faces: torch.Tensor, 
                            num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points on mesh surface using Kaolin
        """
        try:
            points, face_indices = kal.ops.mesh.sample_points(
                vertices.unsqueeze(0), 
                faces, 
                num_samples
            )
            return points.squeeze(0), face_indices.squeeze(0)
        except Exception as e:
            print(f"Point sampling failed: {e}")
            # Fallback: random points on vertices
            point_indices = torch.randint(0, len(vertices), (num_samples,))
            return vertices[point_indices], point_indices
    
    @staticmethod
    def create_voxel_grid(vertices: torch.Tensor, faces: torch.Tensor, 
                         resolution: int = 64) -> torch.Tensor:
        """
        Create voxel grid from mesh using Kaolin
        """
        try:
            voxel_grid = kal.ops.conversions.trianglemesh.to_voxelgrid(
                vertices.unsqueeze(0), 
                faces, 
                resolution
            )
            return voxel_grid.squeeze(0)
        except Exception as e:
            print(f"Voxelization failed: {e}")
            return torch.zeros((resolution, resolution, resolution))
    
    @staticmethod
    def estimate_curvature(vertices: torch.Tensor, faces: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Estimate curvature using Kaolin operations
        This is a simplified curvature estimation
        """
        # Compute vertex normals
        vertex_normals = KaolinHelper.compute_vertex_normals(vertices, faces)
        
        # Simple curvature approximation based on normal variation
        curvature = torch.zeros(len(vertices), device=vertices.device)
        
        # For each vertex, compute average normal difference with neighbors
        for i in range(len(vertices)):
            # Find faces containing this vertex
            face_mask = (faces == i).any(dim=1)
            connected_faces = faces[face_mask]
            
            # Get connected vertices
            connected_vertices = torch.unique(connected_faces)
            connected_vertices = connected_vertices[connected_vertices != i]
            
            if len(connected_vertices) > 0:
                # Compute normal differences
                normal_diffs = 1 - torch.nn.functional.cosine_similarity(
                    vertex_normals[i].unsqueeze(0),
                    vertex_normals[connected_vertices],
                    dim=1
                )
                curvature[i] = torch.mean(normal_diffs)
        
        return {
            'mean_curvature': curvature,
            'gaussian_curvature': curvature,  # Simplified
            'vertex_normals': vertex_normals
        }

def test_kaolin_functionality():
    """
    Test Kaolin functionality
    """
    helper = KaolinHelper()
    
    # Check installation
    checks = helper.check_kaolin_installation()
    print("Kaolin Installation Check:")
    for check_name, status in checks.items():
        print(f"  {check_name}: {status}")
    
    if checks['kaolin_imported']:
        # Test basic operations
        print("\nTesting Kaolin operations...")
        
        # Create a simple mesh
        vertices = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32)
        
        faces = torch.tensor([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1]
        ], dtype=torch.long)
        
        # Test operations
        try:
            face_normals = helper.compute_face_normals(vertices, faces)
            vertex_normals = helper.compute_vertex_normals(vertices, faces)
            face_areas = helper.compute_face_areas(vertices, faces)
            
            print("✓ Basic mesh operations working")
            print(f"  Face normals shape: {face_normals.shape}")
            print(f"  Vertex normals shape: {vertex_normals.shape}")
            print(f"  Face areas shape: {face_areas.shape}")
            
        except Exception as e:
            print(f"✗ Mesh operations failed: {e}")
    
    return checks

if __name__ == "__main__":
    test_kaolin_functionality()
