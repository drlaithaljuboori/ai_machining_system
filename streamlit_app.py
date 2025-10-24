# streamlit_app.py
import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Set page config first
st.set_page_config(
    page_title="AI Machining Analysis System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Main title
    st.title("⚙️ AI-Powered Geometric Analysis & Machining System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "CAD Analysis", "Curvature Classification", "Tool Selection", "Parameter Optimization"]
    )
    
    # Initialize session state for results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if app_mode == "Home":
        show_home()
    elif app_mode == "CAD Analysis":
        show_cad_analysis()
    elif app_mode == "Curvature Classification":
        show_curvature_analysis()
    elif app_mode == "Tool Selection":
        show_tool_selection()
    elif app_mode == "Parameter Optimization":
        show_parameter_optimization()

def show_home():
    """Home page with overview and instructions"""
    st.header("Welcome to AI Machining System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 System Overview
        
        This AI-powered system provides:
        
        - **3D Geometry Analysis** - Automatic CAD file processing
        - **Curvature Classification** - AI-based surface analysis  
        - **Intelligent Tool Selection** - Optimal tool recommendations
        - **Parameter Optimization** - AI-optimized machining parameters
        
        ### 📁 Supported Formats
        - STL, OBJ, STEP, IGES files
        - All major CAD formats via conversion
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/4A90E2/FFFFFF?text=AI+Machining", 
                caption="AI-Powered Manufacturing")
        
        st.info("""
        **Quick Start:**
        1. Upload CAD file
        2. Set material & machine
        3. Run analysis
        4. Get optimized parameters
        """)
    
    st.markdown("---")
    st.subheader("Getting Started")
    
    # Quick analysis section
    st.write("### 🎯 Quick Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload CAD File", 
        type=['stl', 'obj', 'step', 'stp'],
        help="Upload your 3D model file"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_file = Path("temp_upload") / uploaded_file.name
        temp_file.parent.mkdir(exist_ok=True)
        
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        # Quick analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            material = st.selectbox(
                "Material",
                ["Steel", "Aluminum", "Stainless Steel", "Titanium", "Other"]
            )
        
        with col2:
            machine_type = st.selectbox(
                "Machine Type", 
                ["3-Axis", "5-Axis", "High-Speed", "CNC Mill"]
            )
        
        with col3:
            objective = st.selectbox(
                "Optimization Goal",
                ["Balanced", "Maximum MRR", "Best Finish", "Tool Life"]
            )
        
        if st.button("🚀 Run AI Analysis", type="primary"):
            with st.spinner("AI is analyzing your geometry..."):
                try:
                    # Import and run analysis
                    from main import AIMachiningSystem
                    
                    # Initialize system
                    system = AIMachiningSystem()
                    
                    # Run analysis
                    results = system.analyze_part(
                        str(temp_file),
                        material=material.lower(),
                        machine_type=machine_type.lower()
                    )
                    
                    # Store results in session state
                    st.session_state.analysis_results = results
                    
                    st.success("✅ Analysis completed successfully!")
                    st.balloons()
                    
                    # Show quick results
                    show_quick_results(results)
                    
                except Exception as e:
                    st.error(f"❌ Analysis failed: {str(e)}")
                    st.info("💡 Make sure all dependencies are installed correctly.")

def show_quick_results(results):
    """Display quick analysis results"""
    st.subheader("📊 Quick Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'region_classification' in results:
            regions = results['region_classification']
            flat_count = len(regions.get('flat_regions', []))
            st.metric("Flat Regions", f"{flat_count} vertices")
    
    with col2:
        if 'tool_recommendations' in results:
            tools = results['tool_recommendations']
            total_tools = sum(len(op['tools']) for op in tools.get('tool_sequence', []))
            st.metric("Recommended Tools", total_tools)
    
    with col3:
        if 'optimized_parameters' in results:
            params = results['optimized_parameters']
            operations = len(params.get('optimized_operations', {}))
            st.metric("Operations", operations)
    
    with col4:
        if 'optimized_parameters' in results:
            summary = params.get('optimization_summary', {})
            time = summary.get('estimated_machining_time', 'N/A')
            st.metric("Est. Time (min)", time)
    
    # Navigation to detailed views
    st.info("💡 Use the sidebar to view detailed analysis for each section.")

def show_cad_analysis():
    """CAD file analysis interface"""
    st.header("📐 CAD Geometry Analysis")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        show_detailed_geometry_analysis(results)
    else:
        st.warning("⚠️ Please upload a CAD file and run analysis from the Home page first.")
        st.info("Navigate to **Home** to start analysis.")

def show_curvature_analysis():
    """Curvature analysis interface"""
    st.header("📊 Curvature Classification")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        show_detailed_curvature_analysis(results)
    else:
        st.warning("⚠️ Please upload a CAD file and run analysis from the Home page first.")

def show_tool_selection():
    """Tool selection interface"""
    st.header("🔧 Tool Selection")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        show_detailed_tool_selection(results)
    else:
        st.warning("⚠️ Please upload a CAD file and run analysis from the Home page first.")

def show_parameter_optimization():
    """Parameter optimization interface"""
    st.header("⚡ Parameter Optimization")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        show_detailed_parameter_optimization(results)
    else:
        st.warning("⚠️ Please upload a CAD file and run analysis from the Home page first.")

def show_detailed_geometry_analysis(results):
    """Show detailed geometry analysis"""
    st.subheader("Geometry Properties")
    
    if 'mesh_data' in results:
        mesh_data = results['mesh_data']
        vertices = mesh_data.get('vertices', [])
        faces = mesh_data.get('faces', [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Vertices", len(vertices) if hasattr(vertices, '__len__') else 'N/A')
        with col2:
            st.metric("Faces", len(faces) if hasattr(faces, '__len__') else 'N/A')
        with col3:
            st.metric("File Type", "3D Mesh")
        with col4:
            st.metric("Status", "Analyzed")
    
    # Show curvature visualization if available
    if 'curvature_map' in results:
        st.subheader("Curvature Visualization")
        st.info("Curvature analysis completed. View in Curvature Classification section.")

def show_detailed_curvature_analysis(results):
    """Show detailed curvature analysis"""
    st.subheader("Curvature Classification Results")
    
    if 'region_classification' in results:
        regions = results['region_classification']
        
        # Create metrics for each region type
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            flat_count = len(regions.get('flat_regions', []))
            st.metric("Flat Regions", flat_count)
        
        with col2:
            convex_count = len(regions.get('convex_regions', []))
            st.metric("Convex Regions", convex_count)
        
        with col3:
            concave_count = len(regions.get('concave_regions', []))
            st.metric("Concave Regions", concave_count)
        
        with col4:
            high_curve_count = len(regions.get('high_curvature', []))
            st.metric("High Curvature", high_curve_count)
        
        # Show curvature statistics
        if 'curvature_statistics' in regions:
            stats = regions['curvature_statistics']
            st.subheader("Curvature Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean Curvature", f"{stats.get('mean_gaussian', 0):.4f}")
            with col2:
                st.metric("Std Dev", f"{stats.get('std_gaussian', 0):.4f}")
            with col3:
                st.metric("Min Curvature", f"{stats.get('min_curvature', 0):.4f}")
            with col4:
                st.metric("Max Curvature", f"{stats.get('max_curvature', 0):.4f}")

def show_detailed_tool_selection(results):
    """Show detailed tool selection"""
    st.subheader("Tool Recommendations")
    
    if 'tool_recommendations' in results:
        tool_recs = results['tool_recommendations']
        tool_sequence = tool_recs.get('tool_sequence', [])
        
        for i, operation in enumerate(tool_sequence):
            with st.expander(f"Operation {i+1}: {operation['operation'].title()}", expanded=True):
                st.write(f"**Strategy:** {operation.get('strategy', 'Standard').replace('_', ' ').title()}")
                
                # Display tools in a table
                tools = operation.get('tools', [])
                if tools:
                    # Create a simple table
                    for j, tool in enumerate(tools):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.write(f"**Tool {j+1}:** {tool.get('name', 'N/A')}")
                        with col2:
                            st.write(f"Diameter: {tool.get('diameter', 'N/A')}mm")
                        with col3:
                            st.write(f"Type: {tool.get('type', 'N/A').replace('_', ' ').title()}")
                        with col4:
                            st.write(f"Flutes: {tool.get('flutes', 'N/A')}")
                        st.markdown("---")

def show_detailed_parameter_optimization(results):
    """Show detailed parameter optimization"""
    st.subheader("Optimized Machining Parameters")
    
    if 'optimized_parameters' in results:
        optimized = results['optimized_parameters']
        operations = optimized.get('optimized_operations', {})
        
        for op_name, tools in operations.items():
            with st.expander(f"Operation: {op_name.title()}", expanded=True):
                for tool_data in tools:
                    tool_name = tool_data.get('tool', {}).get('name', 'Unknown Tool')
                    st.write(f"**{tool_name}**")
                    
                    # Cutting parameters
                    cutting_params = tool_data.get('cutting_parameters', {})
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("RPM", cutting_params.get('rpm', 'N/A'))
                    with col2:
                        st.metric("Feed Rate", f"{cutting_params.get('feed_rate', 'N/A')} mm/min")
                    with col3:
                        st.metric("Axial DOC", f"{cutting_params.get('axial_depth_of_cut', 'N/A')} mm")
                    with col4:
                        st.metric("Radial DOC", f"{cutting_params.get('radial_depth_of_cut', 'N/A')} mm")
                    
                    # Performance metrics
                    metrics = tool_data.get('performance_metrics', {})
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("MRR", f"{metrics.get('material_removal_rate', 'N/A')} cm³/min")
                    with col2:
                        st.metric("Power", f"{metrics.get('cutting_power', 'N/A')} kW")
                    with col3:
                        st.metric("Tool Life", f"{metrics.get('estimated_tool_life', 'N/A')} min")
                    with col4:
                        roughness = metrics.get('estimated_surface_roughness', 'N/A')
                        st.metric("Surface Roughness", f"{roughness} µm" if roughness != 'N/A' else 'N/A')
                    
                    st.markdown("---")

if __name__ == "__main__":
    main()
# Add to streamlit_app.py
@st.cache_resource
def load_ai_system():
    """Cache the AI system to avoid reloading"""
    from main import AIMachiningSystem
    return AIMachiningSystem()

# Usage
system = load_ai_system()
