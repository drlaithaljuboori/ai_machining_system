# core/parameter_optimizer.py
import torch
import numpy as np
import json
import math
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class MaterialProperties:
    """Material-specific machining properties"""
    name: str
    hardness: float  # HB
    tensile_strength: float  # MPa
    thermal_conductivity: float  # W/m-K
    specific_cutting_force: float  # N/mm²
    recommended_sfm: Dict[str, float]  # Surface Feet per Minute by operation
    recommended_fpt: Dict[str, float]  # Feed per Tooth by operation

class ParameterOptimizer:
    """
    AI-powered machining parameter optimizer
    Uses both physics-based models and ML for parameter optimization
    """
    
    def __init__(self, material_db_path: str = "data/material_database.json"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.material_database = self._load_material_database(material_db_path)
        self.optimization_models = self._initialize_models()
        
    def _load_material_database(self, path: str) -> Dict[str, MaterialProperties]:
        """Load material database from JSON file"""
        try:
            with open(path, 'r') as f:
                material_data = json.load(f)
            
            materials = {}
            for name, props in material_data.items():
                materials[name.lower()] = MaterialProperties(
                    name=name,
                    hardness=props.get('hardness', 200),
                    tensile_strength=props.get('tensile_strength', 500),
                    thermal_conductivity=props.get('thermal_conductivity', 50),
                    specific_cutting_force=props.get('specific_cutting_force', 2000),
                    recommended_sfm=props.get('recommended_sfm', {}),
                    recommended_fpt=props.get('recommended_fpt', {})
                )
            return materials
            
        except FileNotFoundError:
            print(f"Material database not found at {path}, using default database")
            return self._create_default_material_database()
    
    def _create_default_material_database(self) -> Dict[str, MaterialProperties]:
        """Create default material database if none exists"""
        return {
            'steel': MaterialProperties(
                name='Steel 1045',
                hardness=170,
                tensile_strength=585,
                thermal_conductivity=51.9,
                specific_cutting_force=2200,
                recommended_sfm={'roughing': 300, 'finishing': 400},
                recommended_fpt={'roughing': 0.15, 'finishing': 0.08}
            ),
            'aluminum': MaterialProperties(
                name='Aluminum 6061',
                hardness=95,
                tensile_strength=310,
                thermal_conductivity=167,
                specific_cutting_force=800,
                recommended_sfm={'roughing': 800, 'finishing': 1200},
                recommended_fpt={'roughing': 0.2, 'finishing': 0.1}
            ),
            'stainless steel': MaterialProperties(
                name='Stainless Steel 304',
                hardness=170,
                tensile_strength=515,
                thermal_conductivity=16.2,
                specific_cutting_force=2800,
                recommended_sfm={'roughing': 200, 'finishing': 300},
                recommended_fpt={'roughing': 0.12, 'finishing': 0.06}
            ),
            'titanium': MaterialProperties(
                name='Titanium Ti-6Al-4V',
                hardness=334,
                tensile_strength=950,
                thermal_conductivity=6.7,
                specific_cutting_force=3500,
                recommended_sfm={'roughing': 150, 'finishing': 200},
                recommended_fpt={'roughing': 0.08, 'finishing': 0.04}
            )
        }
    
    def _initialize_models(self):
        """Initialize ML models for parameter optimization"""
        # Placeholder for trained ML models
        # In practice, these would be pre-trained models for specific optimization tasks
        models = {
            'surface_roughness_predictor': None,
            'tool_life_predictor': None,
            'power_consumption_predictor': None
        }
        return models
    
    def optimize_parameters(self, tool_recommendations: Dict, material: str, 
                          machine_type: str, objective: str = "balanced") -> Dict:
        """
        Optimize machining parameters for given tools and material
        
        Args:
            tool_recommendations: Output from ToolSelector
            material: Workpiece material
            machine_type: Type of machine (3-axis, 5-axis, etc.)
            objective: Optimization objective ("max_mrr", "best_finish", "tool_life", "balanced")
        
        Returns:
            Dictionary with optimized parameters for each tool and operation
        """
        material_props = self._get_material_properties(material)
        
        optimized_operations = {}
        
        # Optimize parameters for each operation in the sequence
        for operation in tool_recommendations.get('tool_sequence', []):
            op_name = operation['operation']
            tools = operation['tools']
            strategy = operation['strategy']
            
            op_params = []
            for tool in tools:
                tool_params = self._calculate_optimal_parameters(
                    tool, material_props, op_name, strategy, objective, machine_type
                )
                op_params.append(tool_params)
            
            optimized_operations[op_name] = op_params
        
        # Add overall optimization summary
        summary = self._generate_optimization_summary(optimized_operations, objective)
        
        return {
            'optimized_operations': optimized_operations,
            'optimization_summary': summary,
            'material_used': material_props.name,
            'machine_type': machine_type,
            'optimization_objective': objective
        }
    
    def _get_material_properties(self, material: str) -> MaterialProperties:
        """Get material properties with fallback to steel if not found"""
        material_lower = material.lower()
        if material_lower in self.material_database:
            return self.material_database[material_lower]
        else:
            print(f"Material '{material}' not found in database, using steel as default")
            return self.material_database['steel']
    
    def _calculate_optimal_parameters(self, tool: Dict, material: MaterialProperties,
                                   operation: str, strategy: str, objective: str,
                                   machine_type: str) -> Dict:
        """Calculate optimal parameters for a specific tool and operation"""
        
        # Get base parameters from material database
        sfm_base = material.recommended_sfm.get(operation, 300)
        fpt_base = material.recommended_fpt.get(operation, 0.1)
        
        # Apply tool-specific adjustments
        tool_type = tool.get('type', 'end_mill')
        diameter = tool.get('diameter', 10.0)
        flutes = tool.get('flutes', 4)
        
        # Adjust SFM based on tool type and operation
        sfm_adjustment = self._calculate_sfm_adjustment(tool_type, operation, strategy, objective)
        optimized_sfm = sfm_base * sfm_adjustment
        
        # Adjust FPT based on tool type and operation
        fpt_adjustment = self._calculate_fpt_adjustment(tool_type, operation, strategy, objective, diameter)
        optimized_fpt = fpt_base * fpt_adjustment
        
        # Calculate derived parameters
        rpm = self._calculate_rpm(optimized_sfm, diameter)
        feed_rate = self._calculate_feed_rate(rpm, optimized_fpt, flutes)
        mrr = self._calculate_material_removal_rate(feed_rate, operation, diameter)
        cutting_power = self._calculate_cutting_power(mrr, material.specific_cutting_force)
        
        # Calculate depth of cut parameters
        doc_parameters = self._calculate_depth_of_cut(diameter, tool_type, operation, material.hardness)
        
        # Apply machine type constraints
        constrained_params = self._apply_machine_constraints(
            rpm, feed_rate, doc_parameters, machine_type
        )
        
        return {
            'tool': tool,
            'operation': operation,
            'strategy': strategy,
            'cutting_parameters': {
                'sfm': round(optimized_sfm, 1),
                'rpm': round(constrained_params['rpm']),
                'feed_per_tooth': round(optimized_fpt, 4),
                'feed_rate': round(constrained_params['feed_rate'], 1),
                'axial_depth_of_cut': round(constrained_params['axial_doc'], 2),
                'radial_depth_of_cut': round(constrained_params['radial_doc'], 2),
                'stepover': round(constrained_params['radial_doc'] / diameter * 100, 1)
            },
            'performance_metrics': {
                'material_removal_rate': round(mrr, 2),
                'cutting_power': round(cutting_power, 2),
                'estimated_surface_roughness': self._estimate_surface_roughness(optimized_fpt, diameter, tool_type),
                'estimated_tool_life': self._estimate_tool_life(optimized_sfm, material.hardness, operation)
            },
            'optimization_factors': {
                'sfm_adjustment_factor': sfm_adjustment,
                'fpt_adjustment_factor': fpt_adjustment,
                'objective': objective
            }
        }
    
    def _calculate_sfm_adjustment(self, tool_type: str, operation: str, 
                                strategy: str, objective: str) -> float:
        """Calculate SFM adjustment factor based on various factors"""
        adjustment = 1.0
        
        # Operation-based adjustment
        if operation == 'roughing':
            adjustment *= 0.9  # Conservative for roughing
        elif operation == 'finishing':
            adjustment *= 1.1  # Higher for better finish
        
        # Strategy-based adjustment
        if 'high_material_removal' in strategy:
            adjustment *= 0.85
        elif 'high_precision' in strategy:
            adjustment *= 1.15
        
        # Objective-based adjustment
        if objective == "max_mrr":
            adjustment *= 1.2
        elif objective == "tool_life":
            adjustment *= 0.8
        elif objective == "best_finish":
            adjustment *= 1.1
        
        # Tool type adjustment
        if tool_type == 'ball_nose':
            adjustment *= 0.9
        elif tool_type == 'drill':
            adjustment *= 0.7
        
        return max(0.5, min(adjustment, 1.5))  # Clamp between 0.5 and 1.5
    
    def _calculate_fpt_adjustment(self, tool_type: str, operation: str, 
                                strategy: str, objective: str, diameter: float) -> float:
        """Calculate feed per tooth adjustment factor"""
        adjustment = 1.0
        
        # Operation-based adjustment
        if operation == 'roughing':
            adjustment *= 1.2
        elif operation == 'finishing':
            adjustment *= 0.8
        
        # Diameter-based adjustment (smaller tools need lighter feeds)
        if diameter < 3.0:
            adjustment *= 0.7
        elif diameter > 20.0:
            adjustment *= 1.3
        
        # Objective-based adjustment
        if objective == "max_mrr":
            adjustment *= 1.3
        elif objective == "tool_life":
            adjustment *= 0.7
        elif objective == "best_finish":
            adjustment *= 0.6
        
        return max(0.3, min(adjustment, 2.0))
    
    def _calculate_rpm(self, sfm: float, diameter: float) -> float:
        """Calculate RPM from SFM and diameter"""
        if diameter <= 0:
            return 1000.0  # Default safe RPM
        return (sfm * 3.82) / diameter
    
    def _calculate_feed_rate(self, rpm: float, fpt: float, flutes: int) -> float:
        """Calculate feed rate in mm/min"""
        return rpm * fpt * flutes
    
    def _calculate_material_removal_rate(self, feed_rate: float, operation: str, 
                                       diameter: float) -> float:
        """Calculate material removal rate in cm³/min"""
        # Simplified MRR calculation - in practice would use actual DOC values
        if operation == 'roughing':
            depth_factor = 0.3 * diameter
        elif operation == 'semi_finishing':
            depth_factor = 0.15 * diameter
        else:  # finishing
            depth_factor = 0.05 * diameter
        
        return feed_rate * depth_factor * (diameter * 0.5) / 1000  # Convert to cm³/min
    
    def _calculate_cutting_power(self, mrr: float, specific_force: float) -> float:
        """Calculate cutting power in kW"""
        # Power (kW) = MRR (cm³/min) × Specific Cutting Force (N/mm²) × Conversion Factor
        return (mrr * specific_force) / 60000  # Simplified conversion
    
    def _calculate_depth_of_cut(self, diameter: float, tool_type: str, 
                              operation: str, material_hardness: float) -> Dict[str, float]:
        """Calculate recommended depth of cut parameters"""
        
        # Base axial DOC as percentage of diameter
        if operation == 'roughing':
            axial_doc = diameter * 0.5  # 50% of diameter
            radial_doc = diameter * 0.7  # 70% stepover
        elif operation == 'semi_finishing':
            axial_doc = diameter * 0.3   # 30% of diameter
            radial_doc = diameter * 0.4  # 40% stepover
        else:  # finishing
            axial_doc = diameter * 0.1   # 10% of diameter
            radial_doc = diameter * 0.2  # 20% stepover
        
        # Adjust for material hardness
        hardness_factor = max(0.5, min(200 / material_hardness, 1.5))
        axial_doc *= hardness_factor
        radial_doc *= hardness_factor
        
        # Adjust for tool type
        if tool_type == 'ball_nose':
            axial_doc *= 1.2  # Ball nose can handle deeper axial cuts
            radial_doc *= 0.8  # But shallower radial engagement
        
        return {
            'axial_doc': max(0.1, axial_doc),  # Minimum 0.1mm
            'radial_doc': max(0.1, radial_doc)
        }
    
    def _apply_machine_constraints(self, rpm: float, feed_rate: float, 
                                 doc_parameters: Dict, machine_type: str) -> Dict:
        """Apply machine-specific constraints to parameters"""
        
        constrained_rpm = rpm
        constrained_feed = feed_rate
        constrained_axial_doc = doc_parameters['axial_doc']
        constrained_radial_doc = doc_parameters['radial_doc']
        
        # Machine-specific limits
        if machine_type == '3-axis':
            max_rpm = 8000
            max_feed = 5000
        elif machine_type == '5-axis':
            max_rpm = 15000
            max_feed = 10000
        elif 'high_speed' in machine_type.lower():
            max_rpm = 30000
            max_feed = 20000
        else:  # Default conservative limits
            max_rpm = 6000
            max_feed = 4000
        
        # Apply constraints
        constrained_rpm = min(constrained_rpm, max_rpm)
        constrained_feed = min(constrained_feed, max_feed)
        
        return {
            'rpm': constrained_rpm,
            'feed_rate': constrained_feed,
            'axial_doc': constrained_axial_doc,
            'radial_doc': constrained_radial_doc
        }
    
    def _estimate_surface_roughness(self, fpt: float, diameter: float, tool_type: str) -> float:
        """Estimate surface roughness (Ra) in micrometers"""
        # Simplified theoretical roughness calculation
        if tool_type == 'ball_nose':
            base_roughness = fpt * 100  # µm
        else:
            base_roughness = (fpt * fpt * 1000) / (diameter * 8)  # Theoretical Ra
        
        # Add some randomness for realism
        variation = 1.0 + (torch.rand(1).item() * 0.4 - 0.2)  # ±20% variation
        return max(0.1, base_roughness * variation)
    
    def _estimate_tool_life(self, sfm: float, material_hardness: float, operation: str) -> float:
        """Estimate tool life in minutes using Taylor's tool life equation"""
        # Simplified Taylor's equation: VT^n = C
        # Where n ≈ 0.13 for carbide tools, C depends on material and operation
        
        if operation == 'roughing':
            base_life = 45  # minutes
            wear_factor = 2.0
        elif operation == 'semi_finishing':
            base_life = 90
            wear_factor = 1.2
        else:  # finishing
            base_life = 120
            wear_factor = 0.8
        
        # Adjust for material hardness
        hardness_factor = max(0.3, min(200 / material_hardness, 2.0))
        
        # Adjust for cutting speed (Taylor's equation approximation)
        speed_factor = (300 / sfm) ** 0.13  # Reference SFM of 300
        
        estimated_life = base_life * hardness_factor * speed_factor * wear_factor
        return max(10, min(estimated_life, 240))  # Clamp between 10-240 minutes
    
    def _generate_optimization_summary(self, optimized_operations: Dict, objective: str) -> Dict:
        """Generate summary of optimization results"""
        total_mrr = 0
        total_power = 0
        total_machining_time = 0  # Simplified estimation
        operations_count = len(optimized_operations)
        
        for op_name, tools in optimized_operations.items():
            for tool_params in tools:
                total_mrr += tool_params['performance_metrics']['material_removal_rate']
                total_power += tool_params['performance_metrics']['cutting_power']
                # Simplified time estimation (would be more complex in practice)
                total_machining_time += 30  # 30 minutes per tool operation
        
        return {
            'total_operations': operations_count,
            'total_tools': sum(len(tools) for tools in optimized_operations.values()),
            'average_mrr': round(total_mrr / operations_count, 2),
            'total_power_requirement': round(total_power, 2),
            'estimated_machining_time': total_machining_time,
            'optimization_success': True,
            'recommendations': self._generate_recommendations(optimized_operations, objective)
        }
    
    def _generate_recommendations(self, optimized_operations: Dict, objective: str) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []
        
        if objective == "max_mrr":
            recommendations.append("Parameters optimized for maximum material removal rate")
            recommendations.append("Consider using flood coolant for heat management")
        elif objective == "best_finish":
            recommendations.append("Parameters optimized for best surface finish")
            recommendations.append("Consider using finishing-specific toolpaths")
        elif objective == "tool_life":
            recommendations.append("Parameters optimized for extended tool life")
            recommendations.append("Consider using tool wear monitoring")
        else:  # balanced
            recommendations.append("Parameters balanced for efficiency and quality")
            recommendations.append("Monitor tool wear and adjust parameters as needed")
        
        # Add general recommendations
        recommendations.append("Verify parameters with machine operator experience")
        recommendations.append("Consider workpiece clamping and rigidity in final setup")
        
        return recommendations

# Example usage and testing
if __name__ == "__main__":
    # Test the parameter optimizer
    optimizer = ParameterOptimizer()
    
    # Sample tool recommendations (matching ToolSelector output format)
    sample_tools = {
        'tool_sequence': [
            {
                'operation': 'roughing',
                'tools': [
                    {'name': 'EM_16mm_Flat', 'type': 'flat', 'diameter': 16.0, 'flutes': 4},
                    {'name': 'EM_10mm_Flat', 'type': 'flat', 'diameter': 10.0, 'flutes': 4}
                ],
                'strategy': 'high_material_removal'
            },
            {
                'operation': 'finishing',
                'tools': [
                    {'name': 'EM_6mm_Ball', 'type': 'ball_nose', 'diameter': 6.0, 'flutes': 2},
                    {'name': 'EM_3mm_Ball', 'type': 'ball_nose', 'diameter': 3.0, 'flutes': 2}
                ],
                'strategy': 'high_precision'
            }
        ]
    }
    
    # Test optimization
    results = optimizer.optimize_parameters(
        sample_tools, 
        material="aluminum",
        machine_type="5-axis",
        objective="balanced"
    )
    
    print("Parameter Optimization Results:")
    print(json.dumps(results, indent=2, default=str))
