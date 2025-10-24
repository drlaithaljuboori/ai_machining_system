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
                
                # Basic mesh analysis
                vertices = mesh.vertices
                faces = mesh.faces
                
                # Perform analysis
                analysis_results = analyze_mesh_simple(mesh)
                
            # Display results
            display_results(analysis_results, vertices, faces)
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please ensure you uploaded a valid STL file.")

def analyze_mesh_simple(mesh):
    """Simple mesh analysis without Kaolin"""
    results = {}
    
    # Basic geometry properties
    results['vertex_count'] = len(mesh.vertices)
    results['face_count'] = len(mesh.faces)
    results['volume'] = mesh.volume
    results['surface_area'] = mesh.area
    
    # Bounding box
    results['bounding_box'] = mesh.bounds
    results['extents'] = mesh.extents
    
    # Simple curvature approximation
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
        # Use normal variation as simple curvature proxy
        normals = mesh.vertex_normals
        if len(normals) > 0:
            # Calculate normal variation
            curvature_approx = calculate_simple_curvature(normals)
            results['curvature_approx'] = curvature_approx
            
            # Classify regions
            flat_threshold = 0.1
            high_curve_threshold = 0.3
            
            flat_regions = np.where(curvature_approx < flat_threshold)[0]
            high_curvature = np.where(curvature_approx > high_curve_threshold)[0]
            
            results['flat_regions'] = flat_regions
            results['high_curvature'] = high_curvature
    
    # Tool recommendations based on geometry
    results['tool_recommendations'] = recommend_tools_simple(results)
    
    return results

def calculate_simple_curvature(normals):
    """Calculate simple curvature approximation from normals"""
    # For each vertex, calculate average normal difference with neighbors
    curvature = np.zeros(len(normals))
    
    # Simple approach: use normal variation as curvature proxy
    for i in range(len(normals)):
        if i < len(normals) - 1:
            # Calculate dot product between consecutive normals
            dot_product = np.dot(normals[i], normals[i+1])
            curvature[i] = 1 - abs(dot_product)  # 0 = same direction, 1 = opposite
    
    return curvature

def recommend_tools_simple(analysis):
    """Simple tool recommendations based on geometry"""
    tools = []
    
    # Roughing tools based on size
    extents = analysis.get('extents', [1, 1, 1])
    max_extent = max(extents)
    
    if max_extent > 100:  # Large part
        tools.append({"operation": "roughing", "tool": "25mm Face Mill", "reason": "Large part size"})
    elif max_extent > 50:  # Medium part
        tools.append({"operation": "roughing", "tool": "16mm End Mill", "reason": "Medium part size"})
    else:  # Small part
        tools.append({"operation": "roughing", "tool": "10mm End Mill", "reason": "Small part size"})
    
    # Finishing tools based on curvature
    high_curvature_count = len(analysis.get('high_curvature', []))
    total_vertices = analysis.get('vertex_count', 1)
    curvature_ratio = high_curvature_count / total_vertices
    
    if curvature_ratio > 0.3:
        tools.append({"operation": "finishing", "tool": "6mm Ball Nose", "reason": "High curvature areas"})
        tools.append({"operation": "finishing", "tool": "3mm Ball Nose", "reason": "Detailed features"})
    elif curvature_ratio > 0.1:
        tools.append({"operation": "finishing", "tool": "8mm Ball Nose", "reason": "Moderate curvature"})
    else:
        tools.append({"operation": "finishing", "tool": "10mm Flat End Mill", "reason": "Mostly flat surfaces"})
    
    return tools

def display_results(results, vertices, faces):
    """Display analysis results"""
    st.success("✅ Analysis Complete!")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vertices", f"{results['vertex_count']:,}")
    with col2:
        st.metric("Faces", f"{results['face_count']:,}")
    with col3:
        st.metric("Volume", f"{results['volume']:.2f} mm³")
    with col4:
        st.metric("Surface Area", f"{results['surface_area']:.2f} mm²")
    
    # 3D Visualization
    st.subheader("3D Model Visualization")
    fig = create_simple_3d_plot(vertices, faces, results)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tool Recommendations
    st.subheader("Tool Recommendations")
    tools = results.get('tool_recommendations', [])
    
    for tool in tools:
        with st.expander(f"{tool['operation'].title()}: {tool['tool']}"):
            st.write(f"**Reason:** {tool['reason']}")
    
    # Curvature Analysis
    if 'curvature_approx' in results:
        st.subheader("Curvature Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            flat_count = len(results.get('flat_regions', []))
            st.metric("Flat Regions", flat_count)
        
        with col2:
            high_curve_count = len(results.get('high_curvature', []))
            st.metric("High Curvature Regions", high_curve_count)
        
        # Curvature distribution
        curvature_data = results['curvature_approx']
        st.write("Curvature Distribution:")
        st.bar_chart(np.histogram(curvature_data, bins=20)[0])

def create_simple_3d_plot(vertices, faces, results):
    """Create a simple 3D plot without Kaolin"""
    fig = go.Figure()
    
    # Create mesh plot
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    
    # Color by curvature if available
    if 'curvature_approx' in results:
        colors = results['curvature_approx']
        colorscale = 'Viridis'
    else:
        colors = np.ones(len(vertices))
        colorscale = 'Blues'
    
    mesh_plot = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        intensity=colors,
        colorscale=colorscale,
        opacity=0.7,
        name="3D Model"
    )
    
    fig.add_trace(mesh_plot)
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            aspectmode='data'
        ),
        height=500,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    return fig

if __name__ == "__main__":
    main()
