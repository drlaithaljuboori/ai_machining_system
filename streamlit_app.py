# streamlit_app.py
import streamlit as st
import numpy as np
import trimesh
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import sys
import os

# Set page config first
st.set_page_config(
    page_title="AI Machining Analysis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("⚙️ AI Machining Analysis System")
    st.markdown("""
    This system analyzes 3D CAD files and provides machining recommendations.
    Upload your STL file to get started.
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload STL File", 
        type=['stl'],
        help="Upload a 3D STL file for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.stl') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and analyze mesh
            with st.spinner("Analyzing 3D geometry..."):
                mesh = trimesh.load_mesh(tmp_path)
                
                # Perform analysis
                analysis_results = analyze_mesh_simple(mesh)
                
            # Display results
            display_results(analysis_results, mesh)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure you uploaded a valid STL file.")

def analyze_mesh_simple(mesh):
    """Simple mesh analysis without complex dependencies"""
    results = {}
    
    # Basic geometry properties
    results['vertex_count'] = len(mesh.vertices)
    results['face_count'] = len(mesh.faces)
    results['volume'] = mesh.volume
    results['surface_area'] = mesh.area
    
    # Bounding box
    results['bounding_box'] = mesh.bounds
    results['extents'] = mesh.extents
    
    # Simple feature detection
    results['is_watertight'] = mesh.is_watertight
    results['euler_number'] = mesh.euler_number
    
    # Curvature approximation
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        normals = mesh.vertex_normals
        if len(normals) > 0:
            curvature_approx = calculate_simple_curvature(normals)
            results['curvature_approx'] = curvature_approx
            
            # Classify regions
            flat_threshold = 0.1
            high_curve_threshold = 0.3
            
            flat_regions = np.where(curvature_approx < flat_threshold)[0]
            high_curvature = np.where(curvature_approx > high_curve_threshold)[0]
            
            results['flat_regions'] = flat_regions
            results['high_curvature'] = high_curvature
            results['curvature_stats'] = {
                'mean': float(np.mean(curvature_approx)),
                'std': float(np.std(curvature_approx)),
                'max': float(np.max(curvature_approx)),
                'min': float(np.min(curvature_approx))
            }
    
    # Tool recommendations
    results['tool_recommendations'] = recommend_tools_simple(results)
    
    return results

def calculate_simple_curvature(normals):
    """Calculate simple curvature approximation from normals"""
    curvature = np.zeros(len(normals))
    
    # Simple approach: use normal variation
    for i in range(len(normals)):
        if i < len(normals) - 1:
            # Calculate dot product between consecutive normals
            dot_product = np.dot(normals[i], normals[i+1])
            curvature[i] = 1 - abs(dot_product)
        else:
            curvature[i] = curvature[i-1]  # Use previous value for last vertex
    
    return curvature

def recommend_tools_simple(analysis):
    """Simple tool recommendations based on geometry"""
    tools = []
    
    # Roughing tools based on size
    extents = analysis.get('extents', [1, 1, 1])
    max_extent = max(extents)
    
    if max_extent > 100:
        tools.append({
            "operation": "Roughing", 
            "tool": "25mm Face Mill", 
            "reason": "Large part size",
            "parameters": "Low RPM, High Feed"
        })
    elif max_extent > 50:
        tools.append({
            "operation": "Roughing", 
            "tool": "16mm End Mill", 
            "reason": "Medium part size",
            "parameters": "Medium RPM, Medium Feed"
        })
    else:
        tools.append({
            "operation": "Roughing", 
            "tool": "10mm End Mill", 
            "reason": "Small part size",
            "parameters": "High RPM, Low Feed"
        })
    
    # Finishing tools based on curvature
    high_curvature_count = len(analysis.get('high_curvature', []))
    total_vertices = analysis.get('vertex_count', 1)
    curvature_ratio = high_curvature_count / total_vertices
    
    if curvature_ratio > 0.3:
        tools.append({
            "operation": "Finishing", 
            "tool": "6mm Ball Nose", 
            "reason": "High curvature areas",
            "parameters": "High RPM, Fine Stepover"
        })
        tools.append({
            "operation": "Detail", 
            "tool": "3mm Ball Nose", 
            "reason": "Detailed features",
            "parameters": "Very High RPM, Very Fine Stepover"
        })
    elif curvature_ratio > 0.1:
        tools.append({
            "operation": "Finishing", 
            "tool": "8mm Ball Nose", 
            "reason": "Moderate curvature",
            "parameters": "Medium RPM, Fine Stepover"
        })
    else:
        tools.append({
            "operation": "Finishing", 
            "tool": "10mm Flat End Mill", 
            "reason": "Mostly flat surfaces",
            "parameters": "Medium RPM, Medium Stepover"
        })
    
    return tools

def display_results(results, mesh):
    """Display analysis results"""
    st.success("✅ Analysis Complete!")
    
    # Basic metrics
    st.subheader("📊 Geometry Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vertices", f"{results['vertex_count']:,}")
    with col2:
        st.metric("Faces", f"{results['face_count']:,}")
    with col3:
        st.metric("Volume", f"{results['volume']:.2f}")
    with col4:
        st.metric("Surface Area", f"{results['surface_area']:.2f}")
    
    # Additional properties
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Watertight", "Yes" if results['is_watertight'] else "No")
    with col2:
        st.metric("Euler Number", results['euler_number'])
    
    # 3D Visualization
    st.subheader("🎯 3D Model Visualization")
    fig = create_simple_3d_plot(mesh.vertices, mesh.faces, results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Curvature Analysis
    if 'curvature_approx' in results:
        st.subheader("📈 Curvature Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            flat_count = len(results.get('flat_regions', []))
            st.metric("Flat Regions", flat_count)
        
        with col2:
            high_curve_count = len(results.get('high_curvature', []))
            st.metric("High Curvature", high_curve_count)
        
        with col3:
            stats = results.get('curvature_stats', {})
            st.metric("Mean Curvature", f"{stats.get('mean', 0):.3f}")
        
        with col4:
            st.metric("Curvature Std", f"{stats.get('std', 0):.3f}")
        
        # Curvature distribution chart
        curvature_data = results['curvature_approx']
        st.bar_chart(np.histogram(curvature_data, bins=20)[0])
    
    # Tool Recommendations
    st.subheader("🔧 Tool Recommendations")
    tools = results.get('tool_recommendations', [])
    
    for i, tool in enumerate(tools):
        with st.expander(f"{tool['operation']}: {tool['tool']}", expanded=i==0):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Tool Type:**")
                st.write(tool['tool'])
                st.write("**Parameters:**")
                st.info(tool['parameters'])
            with col2:
                st.write("**Reason:**")
                st.success(tool['reason'])
    
    # Export options
    st.subheader("📤 Export Results")
    if st.button("Generate Report Summary"):
        report = generate_report_summary(results)
        st.download_button(
            "Download Report",
            report,
            file_name="machining_analysis_report.txt",
            mime="text/plain"
        )

def create_simple_3d_plot(vertices, faces, results):
    """Create a simple 3D plot"""
    fig = go.Figure()
    
    # Extract coordinates
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    # Color by curvature if available
    if 'curvature_approx' in results:
        colors = results['curvature_approx']
        colorscale = 'Viridis'
        colorbar_title = "Curvature"
    else:
        colors = np.ones(len(vertices))
        colorscale = 'Blues'
        colorbar_title = "Intensity"
    
    # Create mesh plot
    mesh_plot = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=colors,
        colorscale=colorscale,
        opacity=0.8,
        name="3D Model",
        showscale=True,
        colorbar=dict(title=colorbar_title)
    )
    
    fig.add_trace(mesh_plot)
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        title="3D Model with Curvature Analysis"
    )
    
    return fig

def generate_report_summary(results):
    """Generate a simple text report"""
    report = f"""
AI Machining Analysis Report
============================

Geometry Properties:
-------------------
Vertices: {results['vertex_count']:,}
Faces: {results['face_count']:,}
Volume: {results['volume']:.2f} mm³
Surface Area: {results['surface_area']:.2f} mm²
Watertight: {results['is_watertight']}
Euler Number: {results['euler_number']}

Bounding Box:
------------
Extents: {results['extents']}

Curvature Analysis:
------------------
"""
    if 'curvature_stats' in results:
        stats = results['curvature_stats']
        report += f"""
Flat Regions: {len(results.get('flat_regions', []))}
High Curvature Regions: {len(results.get('high_curvature', []))}
Mean Curvature: {stats['mean']:.3f}
Curvature Std: {stats['std']:.3f}
"""

    report += """
Tool Recommendations:
--------------------
"""
    tools = results.get('tool_recommendations', [])
    for tool in tools:
        report += f"""
{tool['operation']}:
  Tool: {tool['tool']}
  Reason: {tool['reason']}
  Parameters: {tool['parameters']}
"""

    return report

if __name__ == "__main__":
    main()
