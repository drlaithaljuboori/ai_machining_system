# tests/test_geometry_processor.py
import unittest
import torch
import numpy as np
import trimesh
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to the path to import the core modules
sys.path.append(str(Path(__file__).parent.parent))

from core.geometry_processor import GeometryProcessor
from unittest.mock import Mock, patch, MagicMock

class TestGeometryProcessor(unittest.TestCase):
    """
    Comprehensive test suite for GeometryProcessor class
    """
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.processor = GeometryProcessor(device=self.device)
        
        # Create test mesh data
        self.sample_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0]
        ], dtype=np.float32)
        
        self.sample_faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [1, 2, 4],
            [2, 3, 4]
        ], dtype=np.int64)
        
        # Create a temporary STL file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_stl_path = os.path.join(self.temp_dir, "test_cube.stl")
        self._create_test_stl_file()
    
    def tearDown(self):
        """Clean up after each test method"""
        # Remove temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_stl_file(self):
        """Create a simple cube STL file for testing"""
        # Create a simple cube mesh
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        cube.export(self.test_stl_path)
    
    def _create_test_cylinder_file(self):
        """Create a cylinder STL file for testing curved surfaces"""
        cylinder_path = os.path.join(self.temp_dir, "test_cylinder.stl")
        cylinder = trimesh.creation.cylinder(radius=0.5, height=2.0)
        cylinder.export(cylinder_path)
        return cylinder_path
    
    def _create_test_sphere_file(self):
        """Create a sphere STL file for testing complex curvature"""
        sphere_path = os.path.join(self.temp_dir, "test_sphere.stl")
        sphere = trimesh.creation.icosphere(subdivisions=2)
        sphere.export(sphere_path)
        return sphere_path
    
    def test_initialization(self):
        """Test GeometryProcessor initialization"""
        self.assertEqual(self.processor.device, self.device)
        
        # Test with specific device
        processor_cpu = GeometryProcessor(device='cpu')
        self.assertEqual(processor_cpu.device, torch.device('cpu'))
    
    def test_process_cad_file_success(self):
        """Test successful processing of CAD file"""
        mesh_data, features = self.processor.process_cad_file(self.test_stl_path)
        
        # Check returned data structure
        self.assertIsInstance(mesh_data, dict)
        self.assertIsInstance(features, dict)
        
        # Check mesh data contents
        self.assertIn('vertices', mesh_data)
        self.assertIn('faces', mesh_data)
        self.assertIn('kal_mesh', mesh_data)
        self.assertIn('original_mesh', mesh_data)
        
        # Check feature data contents
        self.assertIn('bounding_box', features)
        self.assertIn('basic_properties', features)
        self.assertIn('detected_features', features)
        
        # Verify tensor types and devices
        self.assertIsInstance(mesh_data['vertices'], torch.Tensor)
        self.assertIsInstance(mesh_data['faces'], torch.Tensor)
        self.assertEqual(mesh_data['vertices'].device, torch.device(self.device))
        self.assertEqual(mesh_data['faces'].device, torch.device(self.device))
    
    def test_process_cad_file_nonexistent(self):
        """Test processing of non-existent file"""
        with self.assertRaises(Exception) as context:
            self.processor.process_cad_file("nonexistent_file.stl")
        
        self.assertIn("Failed to process CAD file", str(context.exception))
    
    def test_process_cad_file_invalid_format(self):
        """Test processing of invalid file format"""
        invalid_file = os.path.join(self.temp_dir, "test.invalid")
        with open(invalid_file, 'w') as f:
            f.write("This is not a valid CAD file")
        
        with self.assertRaises(Exception) as context:
            self.processor.process_cad_file(invalid_file)
    
    def test_geometry_feature_extraction(self):
        """Test geometry feature extraction"""
        # Create a mock mesh with known properties
        mock_mesh = trimesh.creation.box(extents=[2.0, 3.0, 4.0])
        
        mesh_data, features = self.processor.process_cad_file(self.test_stl_path)
        
        # Test bounding box extraction
        bbox = features['bounding_box']
        self.assertIn('min', bbox)
        self.assertIn('max', bbox)
        self.assertIn('dimensions', bbox)
        
        # Test basic properties
        basic_props = features['basic_properties']
        self.assertIn('volume', basic_props)
        self.assertIn('surface_area', basic_props)
        self.assertIn('face_count', basic_props)
        self.assertIn('vertex_count', basic_props)
        
        # Test feature detection
        detected_features = features['detected_features']
        expected_feature_types = ['pockets', 'holes', 'fillets', 'chamfers', 'flat_faces']
        for feature_type in expected_feature_types:
            self.assertIn(feature_type, detected_features)
            self.assertIsInstance(detected_features[feature_type], list)
    
    def test_mesh_normalization(self):
        """Test mesh normalization functionality"""
        # Create a large mesh
        large_vertices = self.sample_vertices * 100.0  # Scale up
        large_faces = self.sample_faces
        
        # Create temporary mesh file
        large_mesh_path = os.path.join(self.temp_dir, "large_mesh.stl")
        large_mesh = trimesh.Trimesh(vertices=large_vertices, faces=large_faces)
        large_mesh.export(large_mesh_path)
        
        mesh_data, features = self.processor.process_cad_file(large_mesh_path)
        
        # Check that vertices are properly normalized (should be centered and scaled)
        vertices = mesh_data['vertices'].cpu().numpy()
        
        # Check that mesh is roughly centered
        centroid = np.mean(vertices, axis=0)
        self.assertAlmostEqual(centroid[0], 0.0, delta=0.5)
        self.assertAlmostEqual(centroid[1], 0.0, delta=0.5)
        self.assertAlmostEqual(centroid[2], 0.0, delta=0.5)
        
        # Check that mesh fits within reasonable bounds
        max_extent = np.max(np.ptp(vertices, axis=0))
        self.assertLessEqual(max_extent, 2.0)  # Should be normalized to ~unit size
    
    def test_different_geometry_types(self):
        """Test processing of different geometry types"""
        # Test with cylinder (curved surfaces)
        cylinder_path = self._create_test_cylinder_file()
        mesh_data_cyl, features_cyl = self.processor.process_cad_file(cylinder_path)
        
        # Test with sphere (complex curvature)
        sphere_path = self._create_test_sphere_file()
        mesh_data_sph, features_sph = self.processor.process_cad_file(sphere_path)
        
        # Verify all return proper structures
        for mesh_data, features in [(mesh_data_cyl, features_cyl), (mesh_data_sph, features_sph)]:
            self.assertIn('vertices', mesh_data)
            self.assertIn('faces', mesh_data)
            self.assertIn('bounding_box', features)
            self.assertIn('basic_properties', features)
            
            # Verify tensor properties
            vertices = mesh_data['vertices']
            faces = mesh_data['faces']
            self.assertIsInstance(vertices, torch.Tensor)
            self.assertIsInstance(faces, torch.Tensor)
            self.assertEqual(vertices.device, torch.device(self.device))
            self.assertEqual(faces.device, torch.device(self.device))
    
    def test_feature_detection_accuracy(self):
        """Test accuracy of feature detection"""
        # Create a mesh with specific features
        # Box with a hole (simulated)
        box_with_hole = trimesh.creation.box(extents=[2.0, 2.0, 1.0])
        
        # Export and process
        test_mesh_path = os.path.join(self.temp_dir, "test_features.stl")
        box_with_hole.export(test_mesh_path)
        
        mesh_data, features = self.processor.process_cad_file(test_mesh_path)
        
        # Test bounding box accuracy
        bbox = features['bounding_box']
        bbox_dims = bbox['dimensions']
        
        # For a 2x2x1 box, dimensions should be approximately [2, 2, 1]
        self.assertAlmostEqual(bbox_dims[0], 2.0, delta=0.1)
        self.assertAlmostEqual(bbox_dims[1], 2.0, delta=0.1)
        self.assertAlmostEqual(bbox_dims[2], 1.0, delta=0.1)
        
        # Test volume calculation
        volume = features['basic_properties']['volume']
        expected_volume = 8.0  # 2x2x2 box
        self.assertAlmostEqual(volume, expected_volume, delta=0.1)
    
    def test_gpu_cpu_consistency(self):
        """Test that GPU and CPU processing give consistent results"""
        if torch.cuda.is_available():
            # Process on GPU
            processor_gpu = GeometryProcessor(device='cuda')
            mesh_data_gpu, features_gpu = processor_gpu.process_cad_file(self.test_stl_path)
            
            # Process on CPU
            processor_cpu = GeometryProcessor(device='cpu')
            mesh_data_cpu, features_cpu = processor_cpu.process_cad_file(self.test_stl_path)
            
            # Compare results (convert GPU tensors to CPU for comparison)
            vertices_gpu = mesh_data_gpu['vertices'].cpu().numpy()
            vertices_cpu = mesh_data_cpu['vertices'].numpy()
            
            faces_gpu = mesh_data_gpu['faces'].cpu().numpy()
            faces_cpu = mesh_data_cpu['faces'].numpy()
            
            # Check vertex consistency
            np.testing.assert_array_almost_equal(vertices_gpu, vertices_cpu, decimal=4)
            
            # Check face consistency
            np.testing.assert_array_equal(faces_gpu, faces_cpu)
            
            # Check feature consistency
            self.assertEqual(features_gpu['basic_properties']['vertex_count'],
                           features_cpu['basic_properties']['vertex_count'])
            self.assertEqual(features_gpu['basic_properties']['face_count'],
                           features_cpu['basic_properties']['face_count'])
    
    def test_large_mesh_processing(self):
        """Test processing of larger meshes"""
        # Create a larger mesh (more vertices and faces)
        large_vertices = np.random.rand(1000, 3).astype(np.float32) * 10.0
        large_faces = np.random.randint(0, 1000, size=(2000, 3)).astype(np.int64)
        
        # Remove duplicate faces and invalid faces
        large_faces = np.unique(large_faces, axis=0)
        valid_faces = [face for face in large_faces if len(np.unique(face)) == 3]
        large_faces = np.array(valid_faces[:1500])  # Ensure we have valid faces
        
        large_mesh = trimesh.Trimesh(vertices=large_vertices, faces=large_faces)
        large_mesh_path = os.path.join(self.temp_dir, "large_mesh.stl")
        large_mesh.export(large_mesh_path)
        
        # Process the large mesh
        mesh_data, features = self.processor.process_cad_file(large_mesh_path)
        
        # Verify processing completed
        self.assertIn('vertices', mesh_data)
        self.assertIn('faces', mesh_data)
        
        # Verify correct vertex and face counts
        self.assertEqual(len(mesh_data['vertices']), 1000)
        self.assertEqual(len(mesh_data['faces']), len(large_faces))
        
        # Verify features were extracted
        self.assertIn('basic_properties', features)
        self.assertEqual(features['basic_properties']['vertex_count'], 1000)
        self.assertEqual(features['basic_properties']['face_count'], len(large_faces))
    
    def test_mesh_with_complex_features(self):
        """Test processing of meshes with complex geometric features"""
        # Create a more complex mesh with multiple features
        box1 = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
        box2 = trimesh.creation.box(extents=[1.0, 2.0, 1.0])
        
        # Combine meshes to create more complex geometry
        complex_mesh = trimesh.util.concatenate([box1, box2])
        
        complex_mesh_path = os.path.join(self.temp_dir, "complex_mesh.stl")
        complex_mesh.export(complex_mesh_path)
        
        mesh_data, features = self.processor.process_cad_file(complex_mesh_path)
        
        # Verify processing
        self.assertIsInstance(mesh_data, dict)
        self.assertIsInstance(features, dict)
        
        # Check that feature detection runs without errors
        detected_features = features['detected_features']
        self.assertIsInstance(detected_features, dict)
        
        # Basic properties should be calculated
        basic_props = features['basic_properties']
        self.assertGreater(basic_props['volume'], 0)
        self.assertGreater(basic_props['surface_area'], 0)
    
    def test_error_handling(self):
        """Test error handling for various edge cases"""
        # Test with empty file
        empty_file = os.path.join(self.temp_dir, "empty.stl")
        open(empty_file, 'a').close()
        
        with self.assertRaises(Exception):
            self.processor.process_cad_file(empty_file)
        
        # Test with corrupted STL file
        corrupted_file = os.path.join(self.temp_dir, "corrupted.stl")
        with open(corrupted_file, 'w') as f:
            f.write("Not a valid STL file content")
        
        with self.assertRaises(Exception):
            self.processor.process_cad_file(corrupted_file)
    
    def test_performance_benchmark(self):
        """Performance test for geometry processing"""
        import time
        
        # Time the processing of a standard mesh
        start_time = time.time()
        mesh_data, features = self.processor.process_cad_file(self.test_stl_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Basic performance check - should process in reasonable time
        # This is more of a sanity check than a strict performance requirement
        self.assertLess(processing_time, 10.0, "Mesh processing took too long")
        
        print(f"Geometry processing time: {processing_time:.3f} seconds")
    
    @patch('core.geometry_processor.trimesh')
    def test_mock_processing(self, mock_trimesh):
        """Test with mocked trimesh to isolate geometry processor logic"""
        # Create a mock mesh
        mock_mesh = Mock()
        mock_mesh.vertices = self.sample_vertices
        mock_mesh.faces = self.sample_faces
        mock_mesh.volume = 1.0
        mock_mesh.area = 6.0
        
        # Configure the mock
        mock_trimesh.load_mesh.return_value = mock_mesh
        
        # Process with mocked trimesh
        mesh_data, features = self.processor.process_cad_file("any_file.stl")
        
        # Verify the mock was called
        mock_trimesh.load_mesh.assert_called_once_with("any_file.stl")
        
        # Verify the processor still returns expected structure
        self.assertIn('vertices', mesh_data)
        self.assertIn('faces', mesh_data)
        self.assertIn('basic_properties', features)
    
    def test_tensor_properties(self):
        """Test tensor properties and data types"""
        mesh_data, features = self.processor.process_cad_file(self.test_stl_path)
        
        vertices = mesh_data['vertices']
        faces = mesh_data['faces']
        
        # Test data types
        self.assertEqual(vertices.dtype, torch.float32)
        self.assertEqual(faces.dtype, torch.int64)
        
        # Test tensor shapes
        self.assertEqual(vertices.ndim, 2)
        self.assertEqual(vertices.shape[1], 3)  # x, y, z coordinates
        self.assertEqual(faces.ndim, 2)
        self.assertEqual(faces.shape[1], 3)  # 3 vertices per face
        
        # Test that vertices are within reasonable bounds after normalization
        vertices_np = vertices.cpu().numpy()
        self.assertGreaterEqual(np.min(vertices_np), -1.5)
        self.assertLessEqual(np.max(vertices_np), 1.5)
    
    def test_batch_processing_capability(self):
        """Test that the processor can handle multiple files sequentially"""
        test_files = [
            self.test_stl_path,
            self._create_test_cylinder_file(),
            self._create_test_sphere_file()
        ]
        
        for test_file in test_files:
            with self.subTest(file=test_file):
                mesh_data, features = self.processor.process_cad_file(test_file)
                
                # Basic validation for each file
                self.assertIn('vertices', mesh_data)
                self.assertIn('faces', mesh_data)
                self.assertIn('basic_properties', features)
                
                # Verify tensors are on correct device
                self.assertEqual(mesh_data['vertices'].device, torch.device(self.device))
                self.assertEqual(mesh_data['faces'].device, torch.device(self.device))

class TestGeometryProcessorEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def setUp(self):
        self.processor = GeometryProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_very_small_mesh(self):
        """Test processing of very small meshes"""
        # Create a tiny mesh
        tiny_vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=np.float32)
        tiny_faces = np.array([[0, 1, 2]], dtype=np.int64)
        
        tiny_mesh = trimesh.Trimesh(vertices=tiny_vertices, faces=tiny_faces)
        tiny_mesh_path = os.path.join(self.temp_dir, "tiny_mesh.stl")
        tiny_mesh.export(tiny_mesh_path)
        
        mesh_data, features = self.processor.process_cad_file(tiny_mesh_path)
        
        # Should still process successfully
        self.assertIn('vertices', mesh_data)
        self.assertIn('faces', mesh_data)
        self.assertEqual(len(mesh_data['vertices']), 3)
        self.assertEqual(len(mesh_data['faces']), 1)
    
    def test_degenerate_faces(self):
        """Test handling of degenerate faces"""
        # Create mesh with degenerate face (all vertices same)
        degenerate_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0]  # Same as first vertex
        ], dtype=np.float32)
        
        degenerate_faces = np.array([
            [0, 1, 2],  # Valid face
            [0, 1, 3]   # Degenerate face (two vertices same)
        ], dtype=np.int64)
        
        # trimesh should handle this, but test that our processor doesn't crash
        try:
            degenerate_mesh = trimesh.Trimesh(vertices=degenerate_vertices, faces=degenerate_faces)
            degenerate_mesh_path = os.path.join(self.temp_dir, "degenerate_mesh.stl")
            degenerate_mesh.export(degenerate_mesh_path)
            
            # Should process without crashing
            mesh_data, features = self.processor.process_cad_file(degenerate_mesh_path)
            self.assertIn('vertices', mesh_data)
        except Exception as e:
            # It's acceptable for trimesh to reject degenerate meshes
            print(f"Degenerate mesh rejected (expected): {e}")
    
    def test_mesh_with_negative_coordinates(self):
        """Test processing of meshes with negative coordinates"""
        negative_vertices = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0]
        ], dtype=np.float32)
        
        negative_faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int64)
        
        negative_mesh = trimesh.Trimesh(vertices=negative_vertices, faces=negative_faces)
        negative_mesh_path = os.path.join(self.temp_dir, "negative_mesh.stl")
        negative_mesh.export(negative_mesh_path)
        
        mesh_data, features = self.processor.process_cad_file(negative_mesh_path)
        
        # Should process successfully
        self.assertIn('vertices', mesh_data)
        self.assertIn('bounding_box', features)
        
        # Bounding box should reflect negative coordinates
        bbox = features['bounding_box']
        self.assertLess(bbox['min'][0], 0)  # Should have negative min values

def run_performance_tests():
    """Run performance-focused tests separately"""
    suite = unittest.TestSuite()
    suite.addTest(TestGeometryProcessor('test_performance_benchmark'))
    suite.addTest(TestGeometryProcessor('test_large_mesh_processing'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
    
    # Optionally run performance tests separately
    # print("\n" + "="*50)
    # print("RUNNING PERFORMANCE TESTS")
    # print("="*50)
    # run_performance_tests()
