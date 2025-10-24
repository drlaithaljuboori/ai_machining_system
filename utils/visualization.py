# utils/visualization.py
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import torch
import trimesh
import pyvista as pv
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from pathlib import Path
import kaleido  # For static plot export

class VisualizationEngine:
    """
    Comprehensive visualization engine for machining analysis results
    Supports 3D mesh visualization, curvature maps, tool paths, and analysis dashboards
    """
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
        self.color_scales = {
            'curvature': 'RdBu_r',
            'tool_assignment': 'Viridis',
            'machining_time': 'Plasma',
            'material_removal': 'Hot',
            'safety': 'RdYlGn'
        }
        
    def create_curvature_map(self, results: Dict, output_path: str = None) -> go.Figure:
        """
        Create interactive 3D curvature visualization map
        
        Args:
            results: Analysis results from the main system
            output_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        mesh_data = results.get('mesh_data', {})
        curvature_map = results.get('curvature_map', {})
        region_classification = results.get('region_classification', {})
        
        if not mesh_data or 'vertices' not in mesh_data:
            print("Warning: No mesh data available for curvature visualization")
            return self._create_dummy_visualization()
        
        vertices = mesh_data['vertices'].cpu().numpy() if hasattr(mesh_data['vertices'], 'cpu') else mesh_data['vertices']
        faces = mesh_data['faces'].cpu().numpy() if hasattr(mesh_data['faces'], 'cpu') else mesh_data['faces']
        
        # Extract curvature data
        if curvature_map and 'gaussian_curvature' in curvature_map:
            curvature_data = curvature_map['gaussian_curvature']
            if hasattr(curvature_data, 'cpu'):
                curvature_data = curvature_data.cpu().numpy()
        else:
            # Generate synthetic curvature data for demonstration
            curvature_data = self._generate_synthetic_curvature(vertices)
        
        # Create 3D mesh plot
        fig = go.Figure()
        
        # Add mesh with curvature coloring
        self._add_curvature_mesh(fig, vertices, faces, curvature_data, region_classification)
        
        # Add region annotations if available
        if region_classification:
            self._add_region_annotations(fig, region_classification, vertices)
        
        # Update layout
        fig.update_layout(
            title={
                'text': '3D Curvature Analysis & Region Classification',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            template=self.theme,
            coloraxis_colorbar=dict(
                title="Gaussian Curvature",
                titleside="right"
            )
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
            print(f"Curvature map saved to: {output_path}")
        
        return fig
    
    def create_tool_assignment_map(self, results: Dict, output_path: str = None) -> go.Figure:
        """
        Create tool assignment visualization showing which tools are assigned to which regions
        
        Args:
            results: Analysis results from the main system
            output_path: Path to save the visualization
            
        Returns:
            Plotly figure object
        """
        mesh_data = results.get('mesh_data', {})
        tool_recommendations = results.get('tool_recommendations', {})
        region_classification = results.get('region_classification', {})
        
        if not mesh_data or 'vertices' not in mesh_data:
            print("Warning: No mesh data available for tool assignment visualization")
            return self._create_dummy_visualization()
        
        vertices = mesh_data['vertices'].cpu().numpy() if hasattr(mesh_data['vertices'], 'cpu') else mesh_data['vertices']
        faces = mesh_data['faces'].cpu().numpy() if hasattr(mesh_data['faces'], 'cpu') else mesh_data['faces']
        
        # Create tool assignment data
        tool_assignments = self._generate_tool_assignments(vertices, region_classification, tool_recommendations)
        
        fig = go.Figure()
        
        # Add mesh with tool assignment coloring
        self._add_tool_assignment_mesh(fig, vertices, faces, tool_assignments, tool_recommendations)
        
        # Add tool legends
        self._add_tool_legends(fig, tool_recommendations)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Tool Assignment Map & Machining Strategy',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            template=self.theme
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
            print(f"Tool assignment map saved to: {output_path}")
        
        return fig
    
    def create_machining_parameters_dashboard(self, results: Dict, output_path: str = None) -> go.Figure:
        """
        Create comprehensive dashboard showing machining parameters and optimization results
        
        Args:
            results: Analysis results from the main system
            output_path: Path to save the visualization
            
        Returns:
            Plotly figure with subplots
        """
        optimized_params = results.get('optimized_parameters', {})
        tool_recommendations = results.get('tool_recommendations', {})
        
        # Create subplot figure
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Material Removal Rate by Operation',
                'Cutting Parameters Distribution',
                'Tool Life Estimation',
                'Power Consumption Analysis'
            ],
            specs=[
                [{"type": "bar"}, {"type": "box"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. Material Removal Rate by Operation
        self._add_mrr_chart(fig, optimized_params, row=1, col=1)
        
        # 2. Cutting Parameters Distribution
        self._add_parameters_distribution(fig, optimized_params, row=1, col=2)
        
        # 3. Tool Life Estimation
        self._add_tool_life_chart(fig, optimized_params, row=2, col=1)
        
        # 4. Power Consumption Analysis
        self._add_power_analysis(fig, optimized_params, row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Machining Parameters Optimization Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1000,
            width=1400,
            template=self.theme,
            showlegend=True
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
            print(f"Machining dashboard saved to: {output_path}")
        
        return fig
    
    def create_tool_path_simulation(self, results: Dict, output_path: str = None) -> go.Figure:
        """
        Create animated tool path simulation
        
        Args:
            results: Analysis results from the main system
            output_path: Path to save the visualization
            
        Returns:
            Plotly figure with animation
        """
        mesh_data = results.get('mesh_data', {})
        tool_recommendations = results.get('tool_recommendations', {})
        
        if not mesh_data or 'vertices' not in mesh_data:
            print("Warning: No mesh data available for tool path simulation")
            return self._create_dummy_visualization()
        
        vertices = mesh_data['vertices'].cpu().numpy() if hasattr(mesh_data['vertices'], 'cpu') else mesh_data['vertices']
        faces = mesh_data['faces'].cpu().numpy() if hasattr(mesh_data['faces'], 'cpu') else mesh_data['faces']
        
        # Generate simulated tool paths
        tool_paths = self._generate_simulated_tool_paths(vertices, tool_recommendations)
        
        fig = go.Figure()
        
        # Add base mesh
        self._add_base_mesh(fig, vertices, faces)
        
        # Add tool paths
        self._add_tool_paths(fig, tool_paths)
        
        # Add tool animations
        self._add_tool_animations(fig, tool_paths)
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Tool Path Simulation & Machining Strategy',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1200,
            height=800,
            template=self.theme,
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 0}
                    }]
                }]
            }]
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
            print(f"Tool path simulation saved to: {output_path}")
        
        return fig
    
    def create_comparison_analysis(self, original_results: Dict, optimized_results: Dict, 
                                 output_path: str = None) -> go.Figure:
        """
        Create comparison visualization between original and optimized parameters
        
        Args:
            original_results: Results with original/default parameters
            optimized_results: Results with optimized parameters
            output_path: Path to save the visualization
            
        Returns:
            Plotly figure with comparison charts
        """
        fig = sp.make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Material Removal Rate Comparison',
                'Tool Life Comparison',
                'Power Consumption Comparison',
                'Machining Time Comparison'
            ],
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ]
        )
        
        # Extract comparison data
        comparison_data = self._extract_comparison_data(original_results, optimized_results)
        
        # Add comparison charts
        self._add_comparison_bars(fig, comparison_data, 'mrr', 'Material Removal Rate (cm³/min)', 1, 1)
        self._add_comparison_bars(fig, comparison_data, 'tool_life', 'Tool Life (min)', 1, 2)
        self._add_comparison_bars(fig, comparison_data, 'power', 'Power Consumption (kW)', 2, 1)
        self._add_comparison_bars(fig, comparison_data, 'machining_time', 'Machining Time (min)', 2, 2)
        
        fig.update_layout(
            title={
                'text': 'Parameter Optimization Comparison',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=800,
            width=1200,
            template=self.theme,
            showlegend=True
        )
        
        # Save if output path provided
        if output_path:
            fig.write_html(output_path)
            print(f"Comparison analysis saved to: {output_path}")
        
        return fig
    
    def create_export_ready_visualizations(self, results: Dict, output_dir: str = "./output") -> Dict:
        """
        Create all visualizations and save them in various formats
        
        Args:
            results: Complete analysis results
            output_dir: Directory to save all visualizations
            
        Returns:
            Dictionary of saved file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        try:
            # 1. Curvature Map
            curvature_fig = self.create_curvature_map(
                results, 
                f"{output_dir}/curvature_analysis.html"
            )
            saved_files['curvature_map'] = f"{output_dir}/curvature_analysis.html"
            
            # 2. Tool Assignment Map
            tool_fig = self.create_tool_assignment_map(
                results,
                f"{output_dir}/tool_assignment.html"
            )
            saved_files['tool_assignment'] = f"{output_dir}/tool_assignment.html"
            
            # 3. Machining Dashboard
            dashboard_fig = self.create_machining_parameters_dashboard(
                results,
                f"{output_dir}/machining_dashboard.html"
            )
            saved_files['machining_dashboard'] = f"{output_dir}/machining_dashboard.html"
            
            # 4. Tool Path Simulation
            simulation_fig = self.create_tool_path_simulation(
                results,
                f"{output_dir}/tool_path_simulation.html"
            )
            saved_files['tool_path_simulation'] = f"{output_dir}/tool_path_simulation.html"
            
            # 5. Static images for reports
            self._create_static_images(results, output_dir)
            saved_files['static_images'] = f"{output_dir}/static_images"
            
            print(f"All visualizations saved to: {output_dir}")
            
        except Exception as e:
            print(f"Error creating visualizations: {str(e)}")
        
        return saved_files
    
    def _add_curvature_mesh(self, fig: go.Figure, vertices: np.ndarray, faces: np.ndarray,
                          curvature_data: np.ndarray, region_classification: Dict):
        """Add 3D mesh with curvature coloring to figure"""
        
        # Normalize curvature data for coloring
        curvature_normalized = (curvature_data - np.min(curvature_data)) / (np.max(curvature_data) - np.min(curvature_data))
        
        # Create mesh plot
        mesh_plot = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=curvature_normalized,
            colorscale=self.color_scales['curvature'],
            colorbar=dict(
                title="Curvature",
                titleside="right"
            ),
            lighting=dict(
                ambient=0.8,
                diffuse=0.8,
                fresnel=0.1,
                specular=1,
                roughness=0.1
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=1000
            ),
            name="Curvature Map",
            showscale=True
        )
        
        fig.add_trace(mesh_plot)
        
        # Add region highlights if available
        if region_classification and 'high_curvature' in region_classification:
            high_curvature_indices = region_classification['high_curvature']
            if hasattr(high_curvature_indices, 'cpu'):
                high_curvature_indices = high_curvature_indices.cpu().numpy()
            
            if len(high_curvature_indices) > 0:
                high_curvature_vertices = vertices[high_curvature_indices]
                
                scatter_plot = go.Scatter3d(
                    x=high_curvature_vertices[:, 0],
                    y=high_curvature_vertices[:, 1],
                    z=high_curvature_vertices[:, 2],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color='red',
                        symbol='circle'
                    ),
                    name="High Curvature Regions"
                )
                fig.add_trace(scatter_plot)
    
    def _add_tool_assignment_mesh(self, fig: go.Figure, vertices: np.ndarray, faces: np.ndarray,
                                tool_assignments: np.ndarray, tool_recommendations: Dict):
        """Add mesh with tool assignment coloring"""
        
        mesh_plot = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=tool_assignments,
            colorscale=self.color_scales['tool_assignment'],
            colorbar=dict(
                title="Tool Assignment",
                titleside="right"
            ),
            lighting=dict(
                ambient=0.8,
                diffuse=0.8,
                fresnel=0.1,
                specular=1,
                roughness=0.1
            ),
            name="Tool Assignment Map",
            showscale=True
        )
        
        fig.add_trace(mesh_plot)
    
    def _add_tool_legends(self, fig: go.Figure, tool_recommendations: Dict):
        """Add tool legends and annotations"""
        
        tool_sequence = tool_recommendations.get('tool_sequence', [])
        
        # Create legend annotations
        annotations = []
        y_position = 0.95
        
        for i, operation in enumerate(tool_sequence):
            op_name = operation['operation']
            tools = operation['tools']
            
            annotation_text = f"<b>{op_name.title()}:</b><br>"
            for tool in tools[:3]:  # Show first 3 tools
                tool_name = tool.get('name', 'Unknown Tool')
                diameter = tool.get('diameter', 0)
                annotation_text += f"• {tool_name} (Ø{diameter}mm)<br>"
            
            annotations.append(dict(
                x=0.02,
                y=y_position,
                xref="paper",
                yref="paper",
                text=annotation_text,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="left"
            ))
            
            y_position -= 0.15
        
        fig.update_layout(annotations=annotations)
    
    def _add_mrr_chart(self, fig: go.Figure, optimized_params: Dict, row: int, col: int):
        """Add material removal rate chart to subplot"""
        
        operations = optimized_params.get('optimized_operations', {})
        
        operation_names = []
        mrr_values = []
        
        for op_name, tools in operations.items():
            total_mrr = 0
            for tool_params in tools:
                mrr = tool_params.get('performance_metrics', {}).get('material_removal_rate', 0)
                total_mrr += mrr
            
            operation_names.append(op_name.title())
            mrr_values.append(total_mrr)
        
        bar_chart = go.Bar(
            x=operation_names,
            y=mrr_values,
            name="MRR",
            marker_color='lightblue'
        )
        
        fig.add_trace(bar_chart, row=row, col=col)
        fig.update_xaxes(title_text="Operation", row=row, col=col)
        fig.update_yaxes(title_text="MRR (cm³/min)", row=row, col=col)
    
    def _add_parameters_distribution(self, fig: go.Figure, optimized_params: Dict, row: int, col: int):
        """Add cutting parameters distribution box plot"""
        
        operations = optimized_params.get('optimized_operations', {})
        
        rpm_values = []
        feed_values = []
        doc_values = []
        
        for op_name, tools in operations.items():
            for tool_params in tools:
                cutting_params = tool_params.get('cutting_parameters', {})
                rpm_values.append(cutting_params.get('rpm', 0))
                feed_values.append(cutting_params.get('feed_rate', 0))
                doc_values.append(cutting_params.get('axial_depth_of_cut', 0))
        
        # Create box plots
        rpm_box = go.Box(y=rpm_values, name="RPM", boxpoints=False)
        feed_box = go.Box(y=feed_values, name="Feed Rate", boxpoints=False)
        doc_box = go.Box(y=doc_values, name="DOC", boxpoints=False)
        
        fig.add_trace(rpm_box, row=row, col=col)
        fig.add_trace(feed_box, row=row, col=col)
        fig.add_trace(doc_box, row=row, col=col)
        
        fig.update_xaxes(title_text="Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def _add_tool_life_chart(self, fig: go.Figure, optimized_params: Dict, row: int, col: int):
        """Add tool life estimation chart"""
        
        operations = optimized_params.get('optimized_operations', {})
        
        tool_names = []
        tool_lives = []
        
        for op_name, tools in operations.items():
            for tool_params in tools:
                tool_name = tool_params.get('tool', {}).get('name', 'Unknown')
                tool_life = tool_params.get('performance_metrics', {}).get('estimated_tool_life', 0)
                
                tool_names.append(f"{op_name[:3]}_{tool_name}")
                tool_lives.append(tool_life)
        
        bar_chart = go.Bar(
            x=tool_names,
            y=tool_lives,
            name="Tool Life",
            marker_color='lightgreen'
        )
        
        fig.add_trace(bar_chart, row=row, col=col)
        fig.update_xaxes(title_text="Tool", row=row, col=col)
        fig.update_yaxes(title_text="Tool Life (min)", row=row, col=col)
    
    def _add_power_analysis(self, fig: go.Figure, optimized_params: Dict, row: int, col: int):
        """Add power consumption analysis"""
        
        operations = optimized_params.get('optimized_operations', {})
        
        operation_names = []
        power_values = []
        mrr_values = []
        
        for op_name, tools in operations.items():
            total_power = 0
            total_mrr = 0
            
            for tool_params in tools:
                metrics = tool_params.get('performance_metrics', {})
                total_power += metrics.get('cutting_power', 0)
                total_mrr += metrics.get('material_removal_rate', 0)
            
            operation_names.append(op_name.title())
            power_values.append(total_power)
            mrr_values.append(total_mrr)
        
        scatter_plot = go.Scatter(
            x=mrr_values,
            y=power_values,
            mode='markers+text',
            text=operation_names,
            textposition="top center",
            marker=dict(
                size=15,
                color=power_values,
                colorscale='Viridis',
                showscale=True
            ),
            name="Power vs MRR"
        )
        
        fig.add_trace(scatter_plot, row=row, col=col)
        fig.update_xaxes(title_text="MRR (cm³/min)", row=row, col=col)
        fig.update_yaxes(title_text="Power (kW)", row=row, col=col)
    
    def _add_base_mesh(self, fig: go.Figure, vertices: np.ndarray, faces: np.ndarray):
        """Add base mesh to figure"""
        
        mesh_plot = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=0.3,
            name="Workpiece"
        )
        
        fig.add_trace(mesh_plot)
    
    def _add_tool_paths(self, fig: go.Figure, tool_paths: Dict):
        """Add tool paths to figure"""
        
        for tool_name, path_data in tool_paths.items():
            path_lines = go.Scatter3d(
                x=path_data['x'],
                y=path_data['y'],
                z=path_data['z'],
                mode='lines',
                line=dict(
                    width=4,
                    color=path_data['color']
                ),
                name=tool_name
            )
            
            fig.add_trace(path_lines)
    
    def _add_tool_animations(self, fig: go.Figure, tool_paths: Dict):
        """Add tool animation frames"""
        
        frames = []
        
        for tool_name, path_data in tool_paths.items():
            for i in range(len(path_data['x'])):
                frame = go.Frame(
                    data=[
                        *fig.data,  # Keep existing traces
                        go.Scatter3d(
                            x=[path_data['x'][i]],
                            y=[path_data['y'][i]],
                            z=[path_data['z'][i]],
                            mode='markers',
                            marker=dict(
                                size=10,
                                color=path_data['color'],
                                symbol='diamond'
                            ),
                            name=f"{tool_name} Position"
                        )
                    ],
                    name=f"frame_{tool_name}_{i}"
                )
                frames.append(frame)
        
        fig.frames = frames
    
    def _add_comparison_bars(self, fig: go.Figure, comparison_data: Dict, metric: str, 
                           metric_name: str, row: int, col: int):
        """Add comparison bar chart to subplot"""
        
        categories = list(comparison_data.keys())
        original_values = [data[metric]['original'] for data in comparison_data.values()]
        optimized_values = [data[metric]['optimized'] for data in comparison_data.values()]
        
        original_bar = go.Bar(
            name='Original',
            x=categories,
            y=original_values,
            marker_color='lightcoral'
        )
        
        optimized_bar = go.Bar(
            name='Optimized',
            x=categories,
            y=optimized_values,
            marker_color='lightgreen'
        )
        
        fig.add_trace(original_bar, row=row, col=col)
        fig.add_trace(optimized_bar, row=row, col=col)
        
        fig.update_xaxes(title_text="Operation", row=row, col=col)
        fig.update_yaxes(title_text=metric_name, row=row, col=col)
    
    def _generate_synthetic_curvature(self, vertices: np.ndarray) -> np.ndarray:
        """Generate synthetic curvature data for visualization"""
        # Simple curvature-like pattern based on vertex positions
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        curvature = np.sin(x) * np.cos(y) * np.sin(z)
        return curvature
    
    def _generate_tool_assignments(self, vertices: np.ndarray, region_classification: Dict, 
                                 tool_recommendations: Dict) -> np.ndarray:
        """Generate tool assignment data for visualization"""
        tool_assignments = np.zeros(len(vertices))
        
        if not region_classification:
            return tool_assignments
        
        # Assign different values to different region types
        region_mapping = {
            'flat_regions': 1.0,
            'convex_regions': 2.0,
            'concave_regions': 3.0,
            'high_curvature': 4.0
        }
        
        for region_type, value in region_mapping.items():
            if region_type in region_classification:
                indices = region_classification[region_type]
                if hasattr(indices, 'cpu'):
                    indices = indices.cpu().numpy()
                tool_assignments[indices] = value
        
        return tool_assignments
    
    def _generate_simulated_tool_paths(self, vertices: np.ndarray, tool_recommendations: Dict) -> Dict:
        """Generate simulated tool paths for visualization"""
        
        tool_paths = {}
        
        # Get bounding box of the mesh
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        
        # Generate paths for different operations
        operations = tool_recommendations.get('tool_sequence', [])
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, operation in enumerate(operations[:3]):  # Show first 3 operations
            op_name = operation['operation']
            color = colors[i % len(colors)]
            
            # Generate different path patterns based on operation type
            if 'roughing' in op_name.lower():
                path = self._generate_roughing_path(bbox_min, bbox_max)
            elif 'finishing' in op_name.lower():
                path = self._generate_finishing_path(bbox_min, bbox_max)
            else:
                path = self._generate_contour_path(bbox_min, bbox_max)
            
            tool_paths[f"{op_name}_path"] = {
                'x': path[0],
                'y': path[1],
                'z': path[2],
                'color': color
            }
        
        return tool_paths
    
    def _generate_roughing_path(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple:
        """Generate roughing tool path pattern"""
        x = np.linspace(bbox_min[0], bbox_max[0], 20)
        y = np.linspace(bbox_min[1], bbox_max[1], 20)
        z = np.full_like(x, bbox_max[2] - 5)  # Slightly below top surface
        
        # Create zig-zag pattern
        x_path = []
        y_path = []
        z_path = []
        
        for i, yi in enumerate(y):
            if i % 2 == 0:
                x_path.extend(x)
            else:
                x_path.extend(x[::-1])
            y_path.extend([yi] * len(x))
            z_path.extend(z)
        
        return x_path, y_path, z_path
    
    def _generate_finishing_path(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple:
        """Generate finishing tool path pattern"""
        # Spiral pattern for finishing
        t = np.linspace(0, 4 * np.pi, 50)
        r = np.linspace(0, np.min(bbox_max[:2] - bbox_min[:2]) / 2, len(t))
        
        center = (bbox_min[:2] + bbox_max[:2]) / 2
        x = center[0] + r * np.cos(t)
        y = center[1] + r * np.sin(t)
        z = bbox_max[2] - 2 * np.ones_like(x)  # Constant height
        
        return x.tolist(), y.tolist(), z.tolist()
    
    def _generate_contour_path(self, bbox_min: np.ndarray, bbox_max: np.ndarray) -> Tuple:
        """Generate contour tool path pattern"""
        # Contour following pattern
        t = np.linspace(0, 2 * np.pi, 30)
        center = (bbox_min[:2] + bbox_max[:2]) / 2
        radius = np.min(bbox_max[:2] - bbox_min[:2]) / 3
        
        x = center[0] + radius * np.cos(t)
        y = center[1] + radius * np.sin(t)
        z = bbox_max[2] - 3 * np.ones_like(x)
        
        return x.tolist(), y.tolist(), z.tolist()
    
    def _extract_comparison_data(self, original_results: Dict, optimized_results: Dict) -> Dict:
        """Extract data for comparison visualization"""
        
        comparison_data = {}
        
        original_ops = original_results.get('optimized_parameters', {}).get('optimized_operations', {})
        optimized_ops = optimized_results.get('optimized_parameters', {}).get('optimized_operations', {})
        
        for op_name in set(list(original_ops.keys()) + list(optimized_ops.keys())):
            original_tools = original_ops.get(op_name, [])
            optimized_tools = optimized_ops.get(op_name, [])
            
            original_mrr = sum(tool.get('performance_metrics', {}).get('material_removal_rate', 0) 
                             for tool in original_tools)
            optimized_mrr = sum(tool.get('performance_metrics', {}).get('material_removal_rate', 0) 
                              for tool in optimized_tools)
            
            original_life = sum(tool.get('performance_metrics', {}).get('estimated_tool_life', 0) 
                              for tool in original_tools) / max(len(original_tools), 1)
            optimized_life = sum(tool.get('performance_metrics', {}).get('estimated_tool_life', 0) 
                               for tool in optimized_tools) / max(len(optimized_tools), 1)
            
            original_power = sum(tool.get('performance_metrics', {}).get('cutting_power', 0) 
                               for tool in original_tools)
            optimized_power = sum(tool.get('performance_metrics', {}).get('cutting_power', 0) 
                                for tool in optimized_tools)
            
            # Simplified time estimation
            original_time = len(original_tools) * 30
            optimized_time = len(optimized_tools) * 25
            
            comparison_data[op_name] = {
                'mrr': {'original': original_mrr, 'optimized': optimized_mrr},
                'tool_life': {'original': original_life, 'optimized': optimized_life},
                'power': {'original': original_power, 'optimized': optimized_power},
                'machining_time': {'original': original_time, 'optimized': optimized_time}
            }
        
        return comparison_data
    
    def _create_static_images(self, results: Dict, output_dir: str):
        """Create static images for reports"""
        try:
            # Create matplotlib static plots
            self._create_static_plots(results, output_dir)
        except Exception as e:
            print(f"Warning: Could not create static images: {str(e)}")
    
    def _create_static_plots(self, results: Dict, output_dir: str):
        """Create static matplotlib plots"""
        
        optimized_params = results.get('optimized_parameters', {})
        operations = optimized_params.get('optimized_operations', {})
        
        # Create summary bar chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # MRR by operation
        operation_names = []
        mrr_values = []
        
        for op_name, tools in operations.items():
            total_mrr = sum(tool.get('performance_metrics', {}).get('material_removal_rate', 0) 
                          for tool in tools)
            operation_names.append(op_name)
            mrr_values.append(total_mrr)
        
        ax1.bar(operation_names, mrr_values, color='skyblue')
        ax1.set_title('Material Removal Rate by Operation')
        ax1.set_ylabel('MRR (cm³/min)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Tool life
        tool_names = []
        tool_lives = []
        
        for op_name, tools in operations.items():
            for tool in tools:
                tool_name = tool.get('tool', {}).get('name', 'Unknown')
                tool_life = tool.get('performance_metrics', {}).get('estimated_tool_life', 0)
                tool_names.append(f"{op_name[:3]}_{tool_name}")
                tool_lives.append(tool_life)
        
        ax2.bar(tool_names[:8], tool_lives[:8], color='lightgreen')  # Show first 8
        ax2.set_title('Tool Life Estimation')
        ax2.set_ylabel('Tool Life (min)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Cutting parameters
        rpm_values = []
        feed_values = []
        
        for tools in operations.values():
            for tool in tools:
                params = tool.get('cutting_parameters', {})
                rpm_values.append(params.get('rpm', 0))
                feed_values.append(params.get('feed_rate', 0))
        
        ax3.scatter(rpm_values, feed_values, alpha=0.6, color='coral')
        ax3.set_title('Cutting Parameters Distribution')
        ax3.set_xlabel('RPM')
        ax3.set_ylabel('Feed Rate (mm/min)')
        
        # Power consumption
        operation_names = []
        power_values = []
        
        for op_name, tools in operations.items():
            total_power = sum(tool.get('performance_metrics', {}).get('cutting_power', 0) 
                            for tool in tools)
            operation_names.append(op_name)
            power_values.append(total_power)
        
        ax4.bar(operation_names, power_values, color='gold')
        ax4.set_title('Power Consumption by Operation')
        ax4.set_ylabel('Power (kW)')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/machining_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_dummy_visualization(self) -> go.Figure:
        """Create a dummy visualization when no data is available"""
        fig = go.Figure()
        
        fig.add_annotation(
            text="No mesh data available for visualization",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        
        fig.update_layout(
            title="Visualization Not Available",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig

# Example usage and testing
if __name__ == "__main__":
    print("Testing Visualization Engine")
    
    # Create test data
    test_results = {
        'mesh_data': {
            'vertices': np.random.rand(100, 3) * 100,
            'faces': np.random.randint(0, 100, (50, 3))
        },
        'curvature_map': {
            'gaussian_curvature': np.random.randn(100) * 0.1
        },
        'region_classification': {
            'flat_regions': np.array([0, 1, 2, 3]),
            'high_curvature': np.array([10, 11, 12])
        },
        'tool_recommendations': {
            'tool_sequence': [
                {
                    'operation': 'roughing',
                    'tools': [
                        {'name': 'EM_16mm_Flat', 'diameter': 16.0},
                        {'name': 'EM_10mm_Flat', 'diameter': 10.0}
                    ]
                },
                {
                    'operation': 'finishing',
                    'tools': [
                        {'name': 'EM_6mm_Ball', 'diameter': 6.0}
                    ]
                }
            ]
        },
        'optimized_parameters': {
            'optimized_operations': {
                'roughing': [
                    {
                        'tool': {'name': 'EM_16mm_Flat'},
                        'cutting_parameters': {'rpm': 2000, 'feed_rate': 500},
                        'performance_metrics': {
                            'material_removal_rate': 45.5,
                            'estimated_tool_life': 120,
                            'cutting_power': 2.5
                        }
                    }
                ]
            }
        }
    }
    
    # Test visualization engine
    viz_engine = VisualizationEngine()
    
    # Test curvature map
    curvature_fig = viz_engine.create_curvature_map(test_results)
    print("Curvature map created successfully")
    
    # Test tool assignment map
    tool_fig = viz_engine.create_tool_assignment_map(test_results)
    print("Tool assignment map created successfully")
    
    # Test dashboard
    dashboard_fig = viz_engine.create_machining_parameters_dashboard(test_results)
    print("Machining dashboard created successfully")
    
    # Test export
    saved_files = viz_engine.create_export_ready_visualizations(test_results, "./test_output")
    print(f"Test visualizations saved: {list(saved_files.keys())}")
    
    print("All visualization tests completed successfully!")
