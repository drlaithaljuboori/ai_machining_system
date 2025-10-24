# main.py
import torch
import argparse
from core.geometry_processor import GeometryProcessor
from core.curvature_analyzer import CurvatureAnalyzer
from core.tool_selector import ToolSelector
from core.parameter_optimizer import ParameterOptimizer
from utils.visualization import VisualizationEngine
from utils.export import ReportGenerator

class AIMachiningSystem:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.geometry_processor = GeometryProcessor(device)
        self.curvature_analyzer = CurvatureAnalyzer(device)
        self.tool_selector = ToolSelector()
        self.parameter_optimizer = ParameterOptimizer()
        self.visualizer = VisualizationEngine()
        self.reporter = ReportGenerator()
        
        print(f"AI Machining System initialized on device: {device}")
    
    def analyze_part(self, cad_file_path, material="steel", machine_type="3-axis"):
        """
        Main analysis pipeline for a CAD part
        """
        print(f"Analyzing part: {cad_file_path}")
        
        # Step 1: Geometry Processing
        print("Step 1/5: Processing geometry...")
        mesh_data, features = self.geometry_processor.process_cad_file(cad_file_path)
        
        # Step 2: Curvature Analysis
        print("Step 2/5: Analyzing curvature...")
        curvature_map, region_classification = self.curvature_analyzer.analyze(
            mesh_data, features
        )
        
        # Step 3: Tool Selection
        print("Step 3/5: Selecting tools...")
        tool_recommendations = self.tool_selector.select_tools(
            region_classification, material, machine_type
        )
        
        # Step 4: Parameter Optimization
        print("Step 4/5: Optimizing parameters...")
        optimized_params = self.parameter_optimizer.optimize_parameters(
            tool_recommendations, material, machine_type
        )
        
        # Step 5: Generate Results
        print("Step 5/5: Generating report...")
        results = {
            'mesh_data': mesh_data,
            'curvature_map': curvature_map,
            'region_classification': region_classification,
            'tool_recommendations': tool_recommendations,
            'optimized_parameters': optimized_params,
            'material': material,
            'machine_type': machine_type
        }
        
        return results
    
    def generate_report(self, results, output_dir="./output"):
        """Generate comprehensive report and visualizations"""
        # Create visualizations
        self.visualizer.create_curvature_map(results, f"{output_dir}/curvature_map.html")
        self.visualizer.create_tool_assignment_map(results, f"{output_dir}/tool_map.html")
        
        # Generate PDF report
        report_path = self.reporter.generate_pdf_report(results, output_dir)
        
        return report_path

def main():
    parser = argparse.ArgumentParser(description="AI-Powered Machining Analysis System")
    parser.add_argument("--input", "-i", required=True, help="Input CAD file path")
    parser.add_argument("--material", "-m", default="steel", help="Workpiece material")
    parser.add_argument("--machine", "-mc", default="3-axis", help="Machine type")
    parser.add_argument("--output", "-o", default="./output", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize system
    system = AIMachiningSystem()
    
    try:
        # Analyze part
        results = system.analyze_part(
            args.input, 
            material=args.material, 
            machine_type=args.machine
        )
        
        # Generate report
        report_path = system.generate_report(results, args.output)
        
        print(f"Analysis complete! Report generated: {report_path}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
