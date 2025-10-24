# main.py - FastAPI Backend with Kaolin
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import kaolin as kal
import kaolin.ops.mesh as mesh_ops
import trimesh
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Any
import json

app = FastAPI(title="AI Machining Analysis API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIMachiningSystem:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def analyze_stl_file(self, file_path: str, material: str = "aluminum", 
                        machine_type: str = "3axis") -> Dict[str, Any]:
        """Analyze STL file using Kaolin"""
        try:
            # Load mesh with trimesh
            mesh = trimesh.load_mesh(file_path)
            
            # Convert to Kaolin format
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device=self.device)
            faces = torch.tensor(mesh.faces, dtype=torch.long, device=self.device)
            
            # Normalize mesh
            vertices = self.normalize_mesh(vertices)
            
            # Perform analysis
            geometry_analysis = self.analyze_geometry(vertices, faces, mesh)
            curvature_analysis = self.analyze_curvature(vertices, faces)
            tool_recommendations = self.recommend_tools(geometry_analysis, curvature_analysis, material, machine_type)
            
            return {
                "success": True,
                "geometry": geometry_analysis,
                "curvature": curvature_analysis,
                "tools": tool_recommendations,
                "metadata": {
                    "material": material,
                    "machine_type": machine_type,
                    "device_used": self.device
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def normalize_mesh(self, vertices: torch.Tensor) -> torch.Tensor:
        """Normalize mesh to unit sphere"""
        # Center
        centroid = torch.mean(vertices, dim=0)
        vertices = vertices - centroid
        
        # Scale
        max_extent = torch.max(torch.abs(vertices))
        if max_extent > 0:
            vertices = vertices / max_extent
            
        return vertices
    
    def analyze_geometry(self, vertices: torch.Tensor, faces: torch.Tensor, 
                        original_mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyze geometric properties"""
        # Basic properties
        vertex_count = len(vertices)
        face_count = len(faces)
        
        # Compute face normals and areas using Kaolin
        face_normals = mesh_ops.face_normals(vertices.unsqueeze(0), faces)
        face_areas = mesh_ops.face_areas(vertices.unsqueeze(0), faces)
        
        # Bounding box
        bbox_min = torch.min(vertices, dim=0)[0].cpu().numpy()
        bbox_max = torch.max(vertices, dim=0)[0].cpu().numpy()
        bbox_dims = bbox_max - bbox_min
        
        return {
            "vertex_count": vertex_count,
            "face_count": face_count,
            "volume": float(original_mesh.volume),
            "surface_area": float(original_mesh.area),
            "bounding_box": {
                "min": bbox_min.tolist(),
                "max": bbox_max.tolist(),
                "dimensions": bbox_dims.tolist()
            },
            "is_watertight": original_mesh.is_watertight,
            "euler_number": original_mesh.euler_number
        }
    
    def analyze_curvature(self, vertices: torch.Tensor, faces: torch.Tensor) -> Dict[str, Any]:
        """Analyze curvature using Kaolin"""
        # Compute vertex normals
        vertex_normals = mesh_ops.vertex_normals(vertices.unsqueeze(0), faces).squeeze(0)
        
        # Simple curvature estimation based on normal variation
        curvature = self.estimate_curvature_from_normals(vertex_normals)
        
        # Classify regions
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
            },
            "curvature_distribution": curvature_np.tolist()
        }
    
    def estimate_curvature_from_normals(self, normals: torch.Tensor) -> torch.Tensor:
        """Estimate curvature from vertex normals"""
        curvature = torch.zeros(len(normals), device=self.device)
        
        # Simple approach: average normal variation with neighbors
        for i in range(len(normals)):
            if i < len(normals) - 1:
                # Dot product between consecutive normals
                dot_product = torch.dot(normals[i], normals[i+1])
                curvature[i] = 1 - torch.abs(dot_product)
            else:
                curvature[i] = curvature[i-1]
                
        return curvature
    
    def recommend_tools(self, geometry: Dict, curvature: Dict, 
                       material: str, machine_type: str) -> list:
        """Recommend tools based on analysis"""
        tools = []
        
        # Roughing tool based on size
        bbox_dims = geometry["bounding_box"]["dimensions"]
        max_dim = max(bbox_dims)
        
        if max_dim > 100:
            roughing_tool = "25mm Face Mill"
            rpm = 1200
            feed = 400
        elif max_dim > 50:
            roughing_tool = "16mm Carbide End Mill"
            rpm = 2400
            feed = 500
        else:
            roughing_tool = "10mm Carbide End Mill"
            rpm = 3000
            feed = 600
        
        tools.append({
            "operation": "Roughing",
            "tool": roughing_tool,
            "reason": f"Efficient material removal for part size ({max_dim:.1f}mm max dimension)",
            "parameters": {
                "rpm": rpm,
                "feed_rate": f"{feed} mm/min",
                "axial_doc": "2.0 mm",
                "radial_doc": "8.0 mm"
            }
        })
        
        # Finishing tools based on curvature
        curvature_ratio = curvature["high_curvature"] / geometry["vertex_count"]
        
        if curvature_ratio > 0.3:
            tools.extend([
                {
                    "operation": "Semi-Finishing",
                    "tool": "8mm Ball Nose End Mill",
                    "reason": "Handle high curvature areas",
                    "parameters": {
                        "rpm": 4000,
                        "feed_rate": "800 mm/min",
                        "stepover": "0.4 mm"
                    }
                },
                {
                    "operation": "Finishing",
                    "tool": "4mm Ball Nose End Mill",
                    "reason": "Detailed features and tight radii",
                    "parameters": {
                        "rpm": 6000,
                        "feed_rate": "1200 mm/min", 
                        "stepover": "0.15 mm"
                    }
                }
            ])
        elif curvature_ratio > 0.1:
            tools.append({
                "operation": "Finishing",
                "tool": "6mm Ball Nose End Mill", 
                "reason": "Moderate curvature surfaces",
                "parameters": {
                    "rpm": 5000,
                    "feed_rate": "1000 mm/min",
                    "stepover": "0.25 mm"
                }
            })
        else:
            tools.append({
                "operation": "Finishing", 
                "tool": "10mm Flat End Mill",
                "reason": "Mostly flat surfaces",
                "parameters": {
                    "rpm": 3000,
                    "feed_rate": "800 mm/min", 
                    "stepover": "0.5 mm"
                }
            })
        
        # Material-specific adjustments
        material_speeds = {
            "aluminum": 1.2,
            "steel": 1.0,
            "stainless": 0.8,
            "titanium": 0.6
        }
        
        speed_factor = material_speeds.get(material, 1.0)
        for tool in tools:
            if "rpm" in tool["parameters"]:
                tool["parameters"]["rpm"] = int(tool["parameters"]["rpm"] * speed_factor)
        
        return tools

# Initialize the AI system
ai_system = AIMachiningSystem()

@app.get("/")
async def root():
    return {"message": "AI Machining Analysis API", "status": "running"}

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
    return {"status": "healthy", "device": ai_system.device}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
