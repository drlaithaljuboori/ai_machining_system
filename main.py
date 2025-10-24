# main.py - Optimized for Railway/Render
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import torch
import trimesh
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import json

app = FastAPI(
    title="AI Machining Analysis API",
    description="AI-powered geometric analysis and tool recommendation system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIMachiningSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 AI Machining System initialized on {self.device}")
        
        # Try to import Kaolin, but have fallbacks
        try:
            import kaolin as kal
            import kaolin.ops.mesh as mesh_ops
            self.kaolin_available = True
            self.kal = kal
            self.mesh_ops = mesh_ops
            print("✅ Kaolin loaded successfully")
        except ImportError as e:
            self.kaolin_available = False
            print(f"⚠️ Kaolin not available: {e}. Using fallback methods.")
    
    def analyze_stl_file(self, file_path: str, material: str = "aluminum", 
                        machine_type: str = "3axis") -> Dict[str, Any]:
        """Analyze STL file with Kaolin fallbacks"""
        try:
            # Load mesh with trimesh
            mesh = trimesh.load_mesh(file_path)
            
            if self.kaolin_available:
                return self._analyze_with_kaolin(mesh, material, machine_type)
            else:
                return self._analyze_with_fallback(mesh, material, machine_type)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def _analyze_with_kaolin(self, mesh: trimesh.Trimesh, material: str, machine_type: str) -> Dict[str, Any]:
        """Analyze using Kaolin"""
        # Convert to PyTorch tensors
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
        
        # Normalize mesh
        vertices = self._normalize_mesh(vertices)
        
        # Analyze geometry
        geometry_analysis = self._analyze_geometry_kaolin(vertices, faces, mesh)
        curvature_analysis = self._analyze_curvature_kaolin(vertices, faces)
        tool_recommendations = self._recommend_tools(geometry_analysis, curvature_analysis, material, machine_type)
        
        return {
            "success": True,
            "geometry": geometry_analysis,
            "curvature": curvature_analysis,
            "tools": tool_recommendations,
            "metadata": {
                "material": material,
                "machine_type": machine_type,
                "analysis_engine": "kaolin",
                "device_used": self.device
            }
        }
    
    def _analyze_with_fallback(self, mesh: trimesh.Trimesh, material: str, machine_type: str) -> Dict[str, Any]:
        """Analyze using fallback methods (no Kaolin)"""
        # Basic geometry analysis with trimesh
        geometry_analysis = {
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
            "volume": float(mesh.volume),
            "surface_area": float(mesh.area),
            "bounding_box": {
                "min": mesh.bounds[0].tolist(),
                "max": mesh.bounds[1].tolist(),
                "dimensions": (mesh.bounds[1] - mesh.bounds[0]).tolist()
            },
            "is_watertight": mesh.is_watertight,
            "euler_number": mesh.euler_number
        }
        
        # Simple curvature approximation
        curvature_analysis = self._analyze_curvature_fallback(mesh)
        tool_recommendations = self._recommend_tools(geometry_analysis, curvature_analysis, material, machine_type)
        
        return {
            "success": True,
            "geometry": geometry_analysis,
            "curvature": curvature_analysis,
            "tools": tool_recommendations,
            "metadata": {
                "material": material,
                "machine_type": machine_type,
                "analysis_engine": "trimesh_fallback",
                "device_used": "cpu"
            }
        }
    
    def _normalize_mesh(self, vertices: torch.Tensor) -> torch.Tensor:
        """Normalize mesh to unit sphere"""
        centroid = torch.mean(vertices, dim=0)
        vertices = vertices - centroid
        max_extent = torch.max(torch.abs(vertices))
        if max_extent > 0:
            vertices = vertices / max_extent
        return vertices
    
    def _analyze_geometry_kaolin(self, vertices: torch.Tensor, faces: torch.Tensor, 
                               mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyze geometry with Kaolin"""
        face_normals = self.mesh_ops.face_normals(vertices.unsqueeze(0), faces)
        face_areas = self.mesh_ops.face_areas(vertices.unsqueeze(0), faces)
        
        bbox_min = torch.min(vertices, dim=0)[0].cpu().numpy()
        bbox_max = torch.max(vertices, dim=0)[0].cpu().numpy()
        
        return {
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "volume": float(mesh.volume),
            "surface_area": float(mesh.area),
            "bounding_box": {
                "min": bbox_min.tolist(),
                "max": bbox_max.tolist(),
                "dimensions": (bbox_max - bbox_min).tolist()
            },
            "is_watertight": mesh.is_watertight,
            "euler_number": mesh.euler_number
        }
    
    def _analyze_curvature_kaolin(self, vertices: torch.Tensor, faces: torch.Tensor) -> Dict[str, Any]:
        """Analyze curvature with Kaolin"""
        vertex_normals = self.mesh_ops.vertex_normals(vertices.unsqueeze(0), faces).squeeze(0)
        curvature = self._estimate_curvature_from_normals(vertex_normals)
        
        curvature_np = curvature.cpu().numpy()
        flat_threshold = 0.05
        high_curve_threshold = 0.15
        
        flat_regions = np.where(curvature_np < flat_threshold)[0]
        high_curvature = np.where(curvature_np > high_curve_threshold)[0]
        
        return {
            "flat_regions": int(len(flat_regions)),
            "high_curvature": int(len(high_curvature)),
            "statistics": {
                "mean": float(np.mean(curvature_np)),
                "std": float(np.std(curvature_np)),
                "min": float(np.min(curvature_np)),
                "max": float(np.max(curvature_np))
            }
        }
    
    def _analyze_curvature_fallback(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Fallback curvature analysis"""
        if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
            normals = mesh.vertex_normals
            # Simple curvature approximation
            curvature = np.zeros(len(normals))
            for i in range(len(normals) - 1):
                dot_product = np.dot(normals[i], normals[i + 1])
                curvature[i] = 1 - abs(dot_product)
            
            flat_threshold = 0.05
            high_curve_threshold = 0.15
            
            flat_regions = np.where(curvature < flat_threshold)[0]
            high_curvature = np.where(curvature > high_curve_threshold)[0]
            
            return {
                "flat_regions": int(len(flat_regions)),
                "high_curvature": int(len(high_curvature)),
                "statistics": {
                    "mean": float(np.mean(curvature)),
                    "std": float(np.std(curvature)),
                    "min": float(np.min(curvature)),
                    "max": float(np.max(curvature))
                }
            }
        else:
            # Return default values if normals not available
            return {
                "flat_regions": len(mesh.vertices) // 2,
                "high_curvature": len(mesh.vertices) // 10,
                "statistics": {
                    "mean": 0.1,
                    "std": 0.05,
                    "min": 0.0,
                    "max": 0.3
                }
            }
    
    def _estimate_curvature_from_normals(self, normals: torch.Tensor) -> torch.Tensor:
        """Estimate curvature from vertex normals"""
        curvature = torch.zeros(len(normals), device=self.device)
        for i in range(len(normals) - 1):
            dot_product = torch.dot(normals[i], normals[i + 1])
            curvature[i] = 1 - torch.abs(dot_product)
        curvature[-1] = curvature[-2]  # Handle last vertex
        return curvature
    
    def _recommend_tools(self, geometry: Dict, curvature: Dict, 
                        material: str, machine_type: str) -> list:
        """Recommend tools based on analysis"""
        tools = []
        
        # Roughing tool based on size
        bbox_dims = geometry["bounding_box"]["dimensions"]
        max_dim = max(bbox_dims)
        
        if max_dim > 100:
            tools.append({
                "operation": "Roughing",
                "tool": "25mm Face Mill",
                "reason": f"Large part size ({max_dim:.1f}mm max dimension)",
                "parameters": "RPM: 1200, Feed: 400 mm/min, DOC: 2mm"
            })
        elif max_dim > 50:
            tools.append({
                "operation": "Roughing",
                "tool": "16mm Carbide End Mill", 
                "reason": f"Medium part size ({max_dim:.1f}mm max dimension)",
                "parameters": "RPM: 2400, Feed: 500 mm/min, DOC: 1.5mm"
            })
        else:
            tools.append({
                "operation": "Roughing",
                "tool": "10mm Carbide End Mill",
                "reason": f"Small part size ({max_dim:.1f}mm max dimension)", 
                "parameters": "RPM: 3000, Feed: 600 mm/min, DOC: 1mm"
            })
        
        # Finishing tools based on curvature
        curvature_ratio = curvature["high_curvature"] / max(geometry["vertex_count"], 1)
        
        if curvature_ratio > 0.3:
            tools.extend([
                {
                    "operation": "Semi-Finishing",
                    "tool": "8mm Ball Nose End Mill",
                    "reason": "High curvature areas detected",
                    "parameters": "RPM: 4000, Feed: 800 mm/min, Stepover: 0.4mm"
                },
                {
                    "operation": "Finishing", 
                    "tool": "4mm Ball Nose End Mill",
                    "reason": "Detailed features and tight radii",
                    "parameters": "RPM: 6000, Feed: 1200 mm/min, Stepover: 0.15mm"
                }
            ])
        elif curvature_ratio > 0.1:
            tools.append({
                "operation": "Finishing",
                "tool": "6mm Ball Nose End Mill",
                "reason": "Moderate curvature surfaces", 
                "parameters": "RPM: 5000, Feed: 1000 mm/min, Stepover: 0.25mm"
            })
        else:
            tools.append({
                "operation": "Finishing",
                "tool": "10mm Flat End Mill", 
                "reason": "Mostly flat surfaces",
                "parameters": "RPM: 3000, Feed: 800 mm/min, Stepover: 0.5mm"
            })
        
        return tools

# Initialize the AI system
ai_system = AIMachiningSystem()

@app.get("/")
async def root():
    return {
        "message": "AI Machining Analysis API", 
        "status": "running",
        "kaolin_available": ai_system.kaolin_available,
        "device": ai_system.device
    }

@app.post("/analyze")
async def analyze_cad_file(
    file: UploadFile = File(...),
    material: str = "aluminum",
    machine_type: str = "3axis"
):
    """Analyze uploaded CAD file"""
    if not file.filename.lower().endswith('.stl'):
        raise HTTPException(status_code=400, detail="Only STL files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Analyze with AI system
        results = ai_system.analyze_stl_file(tmp_path, material, machine_type)
        
        # Clean up
        os.unlink(tmp_path)
        
        if results["success"]:
            return JSONResponse(content=results)
        else:
            raise HTTPException(status_code=500, detail=results["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "device": ai_system.device,
        "kaolin_available": ai_system.kaolin_available
    }

# Serve frontend files
@app.get("/frontend")
async def serve_frontend():
    return FileResponse('index.html')

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
