import json
import os
import glob

class JSONStructureVisualizer:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        
    def visualize_json_structure(self, experiment_folder):
        """Create a text-based visualization of JSON file structures"""
        try:
            # Find all JSON files
            json_files = glob.glob(os.path.join(experiment_folder, '**/*.json'), recursive=True)
            
            # Create structure report
            structure = ["JSON Files Structure", "=" * 50, ""]
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Add file structure to report
                    rel_path = os.path.relpath(json_file, experiment_folder)
                    structure.append(f"\n{rel_path}")
                    structure.append("-" * len(rel_path))
                    
                    # Add structure representation
                    structure.extend(self._format_structure(data))
                    structure.append("\n")
                    
                except Exception as e:
                    print(f"Error processing {json_file}: {str(e)}")
            
            # Save visualization
            viz_path = os.path.join(self.output_folder, 'json_structure.txt')
            with open(viz_path, 'w') as f:
                f.write('\n'.join(structure))
            
            print(f"JSON structure visualization saved to: {viz_path}")
            
        except Exception as e:
            print(f"Error creating JSON structure visualization: {str(e)}")
    
    def _format_structure(self, data, indent=0):
        """Format JSON structure as text"""
        lines = []
        prefix = '  ' * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}{key}:")
                    lines.extend(self._format_structure(value, indent + 1))
                else:
                    value_type = type(value).__name__
                    lines.append(f"{prefix}{key}: <{value_type}>")
                    
        elif isinstance(data, list):
            if data:
                lines.append(f"{prefix}List of {len(data)} items:")
                if isinstance(data[0], (dict, list)):
                    lines.extend(self._format_structure(data[0], indent + 1))
                else:
                    lines.append(f"{prefix}  <{type(data[0]).__name__}>")
            else:
                lines.append(f"{prefix}Empty list")
                
        return lines

def create_json_structure_visualization(experiment_folder, output_folder):
    """Create visualization of JSON file structure"""
    visualizer = JSONStructureVisualizer(output_folder)
    visualizer.visualize_json_structure(experiment_folder)

if __name__ == "__main__":
    # Example usage
    experiment_folder = "path/to/experiment/folder"
    output_folder = "path/to/output/folder"
    create_json_structure_visualization(experiment_folder, output_folder) 