# core/tool_selector.py
import json
import numpy as np
from typing import Dict, List

class ToolSelector:
    def __init__(self, tool_library_path="data/tool_library.json"):
        self.tool_library = self._load_tool_library(tool_library_path)
        self.selection_rules = self._define_selection_rules()
    
    def _load_tool_library(self, path: str) -> Dict:
        """Load tool library from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Tool library not found at {path}, using default library")
            return self._create_default_tool_library()
    
    def _create_default_tool_library(self) -> Dict:
        """Create a default tool library if none exists"""
        return {
            "end_mills": [
                {"name": "EM_1mm_Ball", "type": "ball_nose", "diameter": 1.0, "corner_radius": 0.5},
                {"name": "EM_3mm_Ball", "type": "ball_nose", "diameter": 3.0, "corner_radius": 1.5},
                {"name": "EM_6mm_Ball", "type": "ball_nose", "diameter": 6.0, "corner_radius": 3.0},
                {"name": "EM_10mm_Flat", "type": "flat", "diameter": 10.0, "corner_radius": 0.0},
                {"name": "EM_16mm_Flat", "type": "flat", "diameter": 16.0, "corner_radius": 0.0},
            ],
            "face_mills": [
                {"name": "FM_50mm", "type": "face_mill", "diameter": 50.0}
            ],
            "drills": [
                {"name": "DR_5mm", "type": "drill", "diameter": 5.0},
                {"name": "DR_10mm", "type": "drill", "diameter": 10.0}
            ]
        }
    
    def _define_selection_rules(self) -> Dict:
        """Define rules for tool selection based on geometry"""
        return {
            "flat_regions": {
                "primary_tool": "face_mill",
                "secondary_tool": "large_flat_end_mill",
                "min_diameter": 10.0
            },
            "high_curvature": {
                "primary_tool": "small_ball_nose",
                "max_diameter": 3.0,
                "constraint": "diameter < smallest_radius * 0.8"
            },
            "concave_regions": {
                "primary_tool": "ball_nose",
                "constraint": "tool_radius < feature_radius"
            },
            "convex_regions": {
                "primary_tool": "ball_nose",
                "secondary_tool": "flat_end_mill"
            }
        }
    
    def select_tools(self, region_classification: Dict, material: str, machine_type: str) -> Dict:
        """Select optimal tools based on curvature analysis"""
        
        recommendations = {
            "roughing_tools": [],
            "semi_finishing_tools": [],
            "finishing_tools": [],
            "tool_sequence": [],
            "rationale": {}
        }
        
        # Select tools based on region types
        if len(region_classification['flat_regions']) > 0:
            flat_tools = self._select_flat_tools(region_classification)
            recommendations['roughing_tools'].extend(flat_tools)
        
        if len(region_classification['high_curvature']) > 0:
            hc_tools = self._select_high_curvature_tools(region_classification)
            recommendations['finishing_tools'].extend(hc_tools)
        
        # Create tool sequence
        recommendations['tool_sequence'] = self._create_tool_sequence(recommendations)
        
        return recommendations
    
    def _select_flat_tools(self, regions: Dict) -> List[Dict]:
        """Select tools for flat regions"""
        return [
            tool for tool in self.tool_library['face_mills'] 
            if tool['diameter'] >= 25.0
        ][:2]  # Return top 2 largest face mills
    
    def _select_high_curvature_tools(self, regions: Dict) -> List[Dict]:
        """Select tools for high curvature regions"""
        return [
            tool for tool in self.tool_library['end_mills'] 
            if tool['type'] == 'ball_nose' and tool['diameter'] <= 3.0
        ]
    
    def _create_tool_sequence(self, recommendations: Dict) -> List[Dict]:
        """Create optimal tool sequence"""
        sequence = []
        
        # Roughing with largest tools
        if recommendations['roughing_tools']:
            sequence.append({
                'operation': 'roughing',
                'tools': recommendations['roughing_tools'],
                'strategy': 'high_material_removal'
            })
        
        # Semi-finishing
        if recommendations.get('semi_finishing_tools'):
            sequence.append({
                'operation': 'semi_finishing',
                'tools': recommendations['semi_finishing_tools'],
                'strategy': 'medium_stepover'
            })
        
        # Finishing with small tools
        if recommendations['finishing_tools']:
            sequence.append({
                'operation': 'finishing',
                'tools': recommendations['finishing_tools'],
                'strategy': 'high_precision'
            })
        
        return sequence
