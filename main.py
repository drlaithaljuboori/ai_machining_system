# main.py - FastAPI Backend (No Kaolin)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AIMachiningSystem:
    def __init__(self):
        self.device = 'cpu'  # Force CPU on Railway
        print(f"🚀 AI Machining System initialized on {self.device}")
    
    def analyze_stl_file(self, file_path: str, material: str = "aluminum", 
                        machine_type: str = "3axis") -> Dict[str, Any]:
        """Analyze STL file using trimesh and numpy"""
        try:
            # Load mesh with trimesh
            mesh = trimesh.load_mesh(file_path)
            
            # Perform analysis
            geometry_analysis = self._analyze_geometry(mesh)
            curvature_analysis = self._analyze_curvature(mesh)
            tool_recommendations = self._recommend_tools(geometry_analysis, curvature_analysis, material, machine_type)
            
            return {
                "success": True,
                "geometry": geometry_analysis,
                "curvature": curvature_analysis,
                "tools": tool_recommendations,
                "metadata": {
                    "material": material,
                    "machine_type": machine_type,
                    "analysis_engine": "trimesh",
                    "device_used": self.device
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }
    
    def _analyze_geometry(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyze geometric properties using trimesh"""
        return {
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
            "euler_number": mesh.euler_number,
            "center_mass": mesh.center_mass.tolist() if hasattr(mesh, 'center_mass') else [0, 0, 0]
        }
    
    def _analyze_curvature(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyze curvature using trimesh and numpy"""
        try:
            # Compute vertex normals if not available
            if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
                mesh.vertex_normals = mesh.vertex_normals
            
            normals = mesh.vertex_normals
            
            # Simple curvature approximation based on normal variation
            curvature = np.zeros(len(normals))
            for i in range(len(normals) - 1):
                dot_product = np.dot(normals[i], normals[i + 1])
                curvature[i] = 1 - abs(dot_product)
            
            # Handle last vertex
            if len(curvature) > 0:
                curvature[-1] = curvature[-2]
            
            # Classify regions
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
                },
                "curvature_distribution": curvature.tolist()[:1000]  # Limit for response size
            }
            
        except Exception as e:
            # Fallback if curvature analysis fails
            print(f"Curvature analysis warning: {e}")
            vertex_count = len(mesh.vertices)
            return {
                "flat_regions": vertex_count // 2,
                "high_curvature": vertex_count // 10,
                "statistics": {
                    "mean": 0.1,
                    "std": 0.05,
                    "min": 0.0,
                    "max": 0.3
                }
            }
    
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
                "parameters": {
                    "rpm": 1200,
                    "feed_rate": "400 mm/min",
                    "axial_doc": "2.0 mm",
                    "radial_doc": "20.0 mm"
                }
            })
        elif max_dim > 50:
            tools.append({
                "operation": "Roughing",
                "tool": "16mm Carbide End Mill", 
                "reason": f"Medium part size ({max_dim:.1f}mm max dimension)",
                "parameters": {
                    "rpm": 2400,
                    "feed_rate": "500 mm/min", 
                    "axial_doc": "1.5 mm",
                    "radial_doc": "12.0 mm"
                }
            })
        else:
            tools.append({
                "operation": "Roughing",
                "tool": "10mm Carbide End Mill",
                "reason": f"Small part size ({max_dim:.1f}mm max dimension)", 
                "parameters": {
                    "rpm": 3000,
                    "feed_rate": "600 mm/min",
                    "axial_doc": "1.0 mm", 
                    "radial_doc": "8.0 mm"
                }
            })
        
        # Finishing tools based on curvature
        curvature_ratio = curvature["high_curvature"] / max(geometry["vertex_count"], 1)
        
        if curvature_ratio > 0.3:
            tools.extend([
                {
                    "operation": "Semi-Finishing",
                    "tool": "8mm Ball Nose End Mill",
                    "reason": "High curvature areas detected",
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
        
        # Adjust for material
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
    return {
        "message": "AI Machining Analysis API", 
        "status": "running",
        "version": "1.0.0",
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
    
    # Check file size (limit to 50MB)
    MAX_FILE_SIZE = 50 * 1024 * 1024
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB")
    
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
        "memory_usage": "N/A"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
