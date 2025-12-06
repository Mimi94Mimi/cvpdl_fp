from .policy import QuestionSample as BaseQuestionSample
from utils import is_none
import shortuuid
import base64
import io
from PIL import Image
import numpy as np
import random
import math
import aiohttp
import traceback
import re
import os
from datetime import datetime

class MCTSNode:
    """MCTS Tree Node Class"""
    def __init__(self, state, parent=None, available_actions=None):
        self.state = state  # Node state
        self.parent = parent  # Parent node
        self.children = {}  # Child nodes
        self.visits = 0  # Visit count
        self.value = 0  # Cumulative reward
        self.leaf_reward = 0  # Reward as leaf node
        # Initialize with untried actions
        self.untried_actions = available_actions.copy() if available_actions else []
        # Store expert information
        self.expert_info = None
        # Store valid area ratio of current image (initial 1.0 means full area is valid)
        self.valid_area_ratio = 1.0
        # Store region coordinates of current image relative to original image
        self.region_coords = state.get('region_coords', (0, 0, state['image_width'], state['image_height']))
        # Additional information storage dictionary
        self.extra_info = {}

class MCTSQuestionSample(BaseQuestionSample):
    def __init__(self, row, args, round_idx=0, enable_logging=False):
        super().__init__(row, args, round_idx)
        # Control whether to write logs
        self.enable_logging = enable_logging
        
        # Get image dimensions
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        self.image_width, self.image_height = img.size
        
        # Create 32x32 blank image
        blank_image = Image.new('RGB', (32, 32), color='white')
        buffered = io.BytesIO()
        blank_image.save(buffered, format="PNG")
        self.blank_image = base64.b64encode(buffered.getvalue()).decode()
        
        # MCTS parameters
        self.max_depth = 4  # Maximum exploration depth
        self.c_puct = 1.0  # PUCT constant
        self.n_simulations = 12  # Simulation count
        self.use_ensemble = True # Whether to use ensemble
        
        # Define action space
        self.actions = [
            "repeat_question",
            "zoom_out"  # New zoom out action
        ]
        
        # Define action prompts
        self.action_prompts = {
            "repeat_question": "Repeat the question.",
            "zoom_out": "Zoom out the region by 1.5x"  # New zoom out action prompt
        }
        
        # Define action executor mapping
        self.action_executors = {
            "repeat_question": self.execute_repeat_question_action,
            "zoom_out": self.execute_zoom_out_action  # New zoom out action executor
        }
        
        # MCTS tree root node
        self.root = None
        
        # List to record all explored nodes
        self.explored_nodes = []
        
        # IoU threshold for node similarity detection (-1 means disabled)
        self.iou_threshold = getattr(args, 'iou_threshold', -1)
        
        # Maximum consecutive repeat_question actions allowed (-1 means disabled)
        self.max_consecutive_repeats = getattr(args, 'max_consecutive_repeats', -1)
        
        # Probability of choosing repeat_question action during expansion
        self.repeat_probability = getattr(args, 'repeat_probability', 0.5)
        
        # Visual expert API
        self.expert_ports = [1]  # Multiple expert ports, corresponding to port number +8000
        self.expert_ports = [port + 8000 for port in self.expert_ports]
        self.expert_base_url = "http://localhost:{}/predict"
        
        # Setup log file for node creation
        self.log_file = None
        if self.enable_logging:
            self.setup_log_file()

    def setup_log_file(self):
        """Setup log file for MCTS node creation"""
        # Create log directory
        log_dir = "./logs/mcts_nodes"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create unique log file name with timestamp and question_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_id = self.row.get('index', 'unknown')
        log_filename = f"mcts_nodes_q{question_id}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)
        
        # Open log file
        self.log_file = open(log_path, 'w', encoding='utf-8')
        self.log_path = log_path
        
        # Write header
        self.write_log("=" * 80)
        self.write_log(f"MCTS Node Creation Log")
        self.write_log(f"Question ID: {question_id}")
        self.write_log(f"Question: {self.row['question']}")
        self.write_log(f"Timestamp: {timestamp}")
        self.write_log(f"Model: {self.args.model_path}")
        self.write_log("=" * 80)
        self.write_log("")
    
    def write_log(self, message):
        """Write message to both log file and stdout"""
        if not self.enable_logging:
            return
        print(message)
        if self.log_file:
            self.log_file.write(message + "\n")
            self.log_file.flush()  # Ensure immediate write
    
    def close_log(self):
        """Close log file"""
        if not self.enable_logging:
            return
        if self.log_file:
            self.write_log("\n" + "=" * 80)
            self.write_log(f"Log saved to: {self.log_path}")
            self.write_log("=" * 80)
            self.log_file.close()
            self.log_file = None

    async def extract_key_objects(self):
        """Extract key objects from question"""
        if 'llava' in self.args.model_path:
            # Use improved extraction method
            # Preprocess question text
            question = self.row['question'].replace('?', '')
            stop_words = ['is', 'in the image', 'IS', 'THE', 'IMAGE','what','color of','there', 'a', 'an', 'How']
            for word in stop_words:
                question = re.sub(r'\b' + word + r'\b', '', question, flags=re.IGNORECASE)
            question = ' '.join(question.split())
            
            # Extract objects
            prompt = f"Task: List objects mentioned in text in List format.\nInput text: {question}\nAction: What objects are mentioned in original text? List separated by commas. For example, from \"person with white trousers on the left or right side of the person in blue\", output \"[\"person with white trousers\", \"person in blue\"]\"."
            response = await self.generate(prompt, self.blank_image, max_tokens=50)
            
            # Try to parse as list format
            try:
                objects = eval(response)
            except:
                response = response.replace('[', '').replace(']', '').replace('"', '')
                objects = response.split(',')
                
            # Filter objects
            filtered_objects = []
            for obj in objects:
                obj = obj.strip().lower()
                if obj in question.lower():
                    filtered_objects.append(obj)
                    
            # If filtered result is empty, use original question
            if not filtered_objects:
                filtered_objects = [question]
                    
            return filtered_objects
            
        else:
            # Use original extraction method
            prompt = f"Task: Extract all objects (including people) with their complete descriptions from the question. For example, from 'Is the person with white trousers on the left or right side of the person in blue?', extract 'person with white trousers' and 'person in blue'.\nQuestion: {self.row['question']}\nAction: Only list the objects separated by commas."
            response = await self.generate(prompt, self.blank_image, max_tokens=50)
            
            if "object" in response.lower() or "description" in response.lower():
                objects = response.split()[-1].lower()

            # Split response into list and strip whitespace
            objects = [obj.strip() for obj in response.split(',')]
            return objects
        
    async def get_expert_boxes(self, image, text):
        """Call visual expert to get boxes"""
        try:
            # Randomly select expert
            port = random.choice(self.expert_ports)
            
            expert_url = self.expert_base_url.format(port)
            timeout = aiohttp.ClientTimeout(total=10000)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    expert_url,
                    json={
                        "image": image,  # image is already base64 string
                        "text": text
                    }
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        print(f"Visual expert API returned error status: {response.status}")
                        print(f"Error message: {error_text}")
                        print(f"Request URL: {expert_url}")
                        print(f"Request text: {text}")
                        return None
        except Exception as e:
            print(f"Error calling visual expert: {str(e)}")
            print(f"Request URL: {expert_url}")
            print(f"Request text: {text}")
            print(f"Exception stack: {traceback.format_exc()}")
            return None

    def selection(self, node):
        """Selection phase: Use UCB algorithm to select best child node"""
        # If node has untried actions, return current node for expansion
        if node.untried_actions:
            return node
            
        if not node.children:
            return node
            
        total_visits = sum(child.visits for child in node.children.values())
        
        def ucb_score(child):
            exploit = child.value / child.visits if child.visits > 0 else 0
            explore = math.sqrt(2 * math.log(total_visits) / (child.visits + 1e-8))
            return exploit + self.c_puct * explore
            
        best_child = max(node.children.values(), key=ucb_score)
        return self.selection(best_child)

    async def execute_repeat_question_action(self, node):
        """Execute repeat question action"""
        node_text = self.row['question']
        expert_result = await self.get_expert_boxes(node.state['image'], node_text)
        
        # If expert result contains boxes
        if expert_result and expert_result.get('boxes'):
            # Convert all boxes to numpy array
            boxes = np.array(expert_result['boxes'])
            
            # Calculate union of all boxes
            x1 = np.min(boxes[:, 0])
            y1 = np.min(boxes[:, 1]) 
            x2 = np.max(boxes[:, 2])
            y2 = np.max(boxes[:, 3])
            
            # Add some padding
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(node.state['image_width'], x2 + padding)
            y2 = min(node.state['image_height'], y2 + padding)
            
            # Calculate new valid area ratio
            new_area = (x2 - x1) * (y2 - y1)
            total_area = node.state['image_width'] * node.state['image_height']
            valid_area_ratio = new_area / total_area
            
            # Crop image
            image_bytes = base64.b64decode(node.state['image'])
            img = Image.open(io.BytesIO(image_bytes))
            cropped_img = img.crop((x1, y1, x2, y2))
            
            # Convert cropped image back to base64
            buffered = io.BytesIO()
            cropped_img.save(buffered, format="PNG")
            cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Update region_coords, considering parent node's coordinate offset
            parent_x1, parent_y1, _, _ = node.state['region_coords']
            new_region_coords = (
                parent_x1 + x1,
                parent_y1 + y1,
                parent_x1 + x2,
                parent_y1 + y2
            )
        else:
            # If no boxes obtained, use original image and region
            cropped_image_base64 = node.state['image']
            valid_area_ratio = node.valid_area_ratio
            new_region_coords = node.state['region_coords']
        
        # Create new state
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["repeat_question"]],
            'text': node_text,
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': new_region_coords
        }
        
        # Create new node
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
        child.valid_area_ratio = valid_area_ratio
        
        # Log node information
        self.write_log("\n" + "="*80)
        self.write_log(f"[NEW NODE - repeat_question] Created at depth {child.state['depth']}")
        self.write_log(f"Action: {self.action_prompts['repeat_question']}")
        self.write_log(f"Valid Area Ratio: {child.valid_area_ratio:.4f}")
        self.write_log(f"Region Coords: {child.state['region_coords']}")
        self.write_log(f"Expert Boxes Found: {len(expert_result.get('boxes', [])) if expert_result else 0}")
        self.write_log(f"Action History: {' -> '.join(child.state['action_history'])}")
        self.write_log(f"Node Text: {child.state['text']}")
        self.write_log("="*80 + "\n")
        
        return child

    async def execute_zoom_out_action(self, node):
        """Execute zoom out action on region"""
        # Get current region coordinates
        x1, y1, x2, y2 = node.state['region_coords']
        
        # Calculate region center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Calculate current region width and height
        width = x2 - x1
        height = y2 - y1
        
        # Zoom out by 1.5x
        new_width = width * 1.5
        new_height = height * 1.5
        
        # Calculate new region coordinates
        new_x1 = max(0, center_x - new_width/2)
        new_y1 = max(0, center_y - new_height/2)
        new_x2 = min(node.state['image_width'], center_x + new_width/2)
        new_y2 = min(node.state['image_height'], center_y + new_height/2)
        
        # Crop original image
        image_bytes = base64.b64decode(self.image)
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop((new_x1, new_y1, new_x2, new_y2))
        
        # Convert cropped image to base64
        buffered = io.BytesIO()
        cropped_img.save(buffered, format="PNG")
        cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        final_x1, final_y1, final_x2, final_y2 = new_x1, new_y1, new_x2, new_y2
        
        # If parent node has missing_objects, try to find them in zoomed out region
        if 'missing_objects' in node.state and node.state['missing_objects']:
            missing_objects_text = ', '.join(node.state['missing_objects'])
            expert_result = await self.get_expert_boxes(cropped_image_base64, missing_objects_text)
            
            # If boxes found, calculate union
            if expert_result and expert_result.get('boxes'):
                boxes = np.array(expert_result['boxes'])
                # Calculate union of expert boxes
                expert_x1 = np.min(boxes[:, 0]) + new_x1
                expert_y1 = np.min(boxes[:, 1]) + new_y1
                expert_x2 = np.max(boxes[:, 2]) + new_x1
                expert_y2 = np.max(boxes[:, 3]) + new_y1
                
                # Calculate union with parent node's region
                final_x1 = min(x1, expert_x1)
                final_y1 = min(y1, expert_y1)
                final_x2 = max(x2, expert_x2)
                final_y2 = max(y2, expert_y2)
                
                # Re-crop image
                cropped_img = img.crop((final_x1, final_y1, final_x2, final_y2))
                buffered = io.BytesIO()
                cropped_img.save(buffered, format="PNG")
                cropped_image_base64 = base64.b64encode(buffered.getvalue()).decode()
        else:
            expert_result = await self.get_expert_boxes(cropped_image_base64, ", ".join(self.key_objects))
        
        # Create new state
        new_state = {
            'depth': node.state['depth'] + 1,
            'image': cropped_image_base64,
            'action_history': node.state['action_history'] + [self.action_prompts["zoom_out"]],
            'text': node.state['text'],
            'image_width': node.state['image_width'],
            'image_height': node.state['image_height'],
            'region_coords': (final_x1, final_y1, final_x2, final_y2)
        }
        
        # Create new node
        child = MCTSNode(new_state, parent=node, available_actions=self.actions)
        child.expert_info = expert_result
            
        # Calculate new valid area ratio
        new_area = (final_x2 - final_x1) * (final_y2 - final_y1)
        total_area = node.state['image_width'] * node.state['image_height']
        child.valid_area_ratio = new_area / total_area
        
        # Log node information
        self.write_log("\n" + "="*80)
        self.write_log(f"[NEW NODE - zoom_out] Created at depth {child.state['depth']}")
        self.write_log(f"Action: {self.action_prompts['zoom_out']}")
        self.write_log(f"Valid Area Ratio: {child.valid_area_ratio:.4f}")
        self.write_log(f"Region Coords: {child.state['region_coords']}")
        self.write_log(f"Original Region: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")
        self.write_log(f"Zoomed Region: ({new_x1:.1f}, {new_y1:.1f}, {new_x2:.1f}, {new_y2:.1f})")
        self.write_log(f"Final Region: ({final_x1:.1f}, {final_y1:.1f}, {final_x2:.1f}, {final_y2:.1f})")
        self.write_log(f"Expert Boxes Found: {len(expert_result.get('boxes', [])) if expert_result else 0}")
        self.write_log(f"Missing Objects: {node.state.get('missing_objects', [])}")
        self.write_log(f"Action History: {' -> '.join(child.state['action_history'])}")
        self.write_log("="*80 + "\n")
        
        return child

    def count_consecutive_repeats(self, node):
        """Count consecutive repeat_question actions in the path to this node"""
        count = 0
        current = node
        
        while current is not None and current.state.get('action_history'):
            action_history = current.state['action_history']
            if action_history:
                last_action = action_history[-1]
                if "Repeat" in last_action or "repeat" in last_action:
                    count += 1
                    current = current.parent
                else:
                    # Stop counting when we hit a non-repeat action
                    break
            else:
                break
        
        return count
    
    async def expansion(self, node):
        """Expansion phase: add a new child node"""
        if node.state['depth'] >= self.max_depth or not node.untried_actions:
            return node
        
        # Check if max consecutive repeats limit is enabled and would be exceeded
        if self.max_consecutive_repeats > 0:
            consecutive_repeats = self.count_consecutive_repeats(node)
            
            # If we've reached the limit, remove repeat_question from untried_actions
            if consecutive_repeats >= self.max_consecutive_repeats:
                if "repeat_question" in node.untried_actions:
                    node.untried_actions.remove("repeat_question")
                    self.write_log(f"⚠️  [REPEAT LIMIT] Node at depth {node.state['depth']} has {consecutive_repeats} consecutive repeat_question actions.")
                    self.write_log(f"   Removing repeat_question from available actions (limit: {self.max_consecutive_repeats}).")
                
                # If no actions left, return current node
                if not node.untried_actions:
                    return node
        
        # Choose action based on repeat_probability
        if len(node.untried_actions) > 1 and "repeat_question" in node.untried_actions:
            # If both actions are available, use probability to decide
            if random.random() < self.repeat_probability:
                action = "repeat_question"
            else:
                # Choose from non-repeat actions
                other_actions = [a for a in node.untried_actions if a != "repeat_question"]
                action = random.choice(other_actions)
        else:
            # If only one action available or repeat_question not available, choose randomly
            action = random.choice(node.untried_actions)
        
        node.untried_actions.remove(action)
        
        # Call corresponding action executor
        child = await self.action_executors[action](node)
        node.children[action] = child
        
        return child

    async def simulation(self, node):
        """Simulation phase: execute actions and obtain rewards"""    
        # Get key objects
        key_objects = self.key_objects
        
        # Ask about each key object individually
        all_objects_present = True
        confirmed_objects = []
        missing_objects = []
        for obj in key_objects:
            # Generate question asking if object is in image
            prompt = f"Task: Only answer yes or no.\nQuestion: Is there a {obj} in this image?"
            response = await self.generate(prompt, node.state['image'], max_tokens=10)
            
            # If object is present, add to confirmed list
            if 'yes' in response.lower():
                confirmed_objects.append(obj)
            else:
                missing_objects.append(obj)
                all_objects_present = False
                break
                
        # Record confirmed and missing objects
        node.state['caption'] = ', '.join(confirmed_objects)
        node.state['missing_objects'] = missing_objects
        
        # Only give reward if all key objects are present
        if all_objects_present:
            # Reward is inversely proportional to valid area ratio
            reward = 1 - node.valid_area_ratio
        else:
            reward = 0
        
        # Create node record object for explored_nodes list after simulation
        # Determine action type based on action history
        action_type = "unknown"
        if node.state.get('action_history'):
            last_action = node.state['action_history'][-1]
            if "Repeat" in last_action or "repeat" in last_action:
                action_type = "repeat_question"
            elif "Zoom" in last_action or "zoom" in last_action:
                action_type = "zoom_out"
        
        node_record = {
            "question_id": self.row.get('index', 'unknown'),
            "action_type": action_type,
            "depth": node.state.get('depth', 0),
            "valid_area_ratio": node.valid_area_ratio,
            "region_coords": node.region_coords,
            "expert_boxes_found": len(node.expert_info.get('boxes', [])) if node.expert_info else 0,
            "action_history": node.state.get('action_history', []).copy(),
            "node_text": node.state.get('text', ''),
            "expert_info": node.expert_info,
            "caption": node.state.get('caption', ''),
            "missing_objects": node.state.get('missing_objects', []).copy(),
            "confirmed_objects": confirmed_objects.copy(),
            "all_objects_present": all_objects_present,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
            "parent_depth": node.parent.state.get('depth', -1) if node.parent else -1,
        }
        
        # Check if this node is similar to any existing explored node (only if iou_threshold >= 0)
        is_duplicate = False
        if self.iou_threshold >= 0:
            for existing_record in self.explored_nodes:
                if self.is_similar_node(node_record, existing_record, iou_threshold=self.iou_threshold):
                    is_duplicate = True
                    self.write_log(f"⚠️  [DUPLICATE NODE DETECTED] Node at depth {node.state.get('depth', 0)} is similar to an existing node.")
                    self.write_log(f"   Region IoU >= {self.iou_threshold} and objects match. Preventing further expansion.")
                    break
            
            # If node is a duplicate, mark it as unable to expand by clearing untried actions
            if is_duplicate:
                node.untried_actions = []
                self.write_log(f"   🚫 Node expansion blocked - untried_actions cleared.")
        
        # Add to explored nodes list
        self.explored_nodes.append(node_record)
            
        return reward

    def backpropagation(self, node, reward):
        """Backpropagation phase: update node values"""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
        
    async def single_run(self, root_state):
        """Single MCTS run"""
        if not self.root:
            # Create temporary root node
            temp_root = MCTSNode(root_state, available_actions=self.actions)
            self.write_log("\n" + "🌳"*40)
            self.write_log("CREATING ROOT NODE...")
            self.write_log(f"Question: {root_state['text']}")
            self.write_log(f"Image Size: {root_state['image_width']} x {root_state['image_height']}")
            self.write_log(f"Key Objects: {self.key_objects}")
            self.write_log("🌳"*40 + "\n")
            # Execute repeat_question_action to get real root node
            self.root = await self.execute_repeat_question_action(temp_root)
            self.root.parent = None
            
        # 1. Selection
        node = self.selection(self.root)
        
        if node.state['depth'] >= self.max_depth:
            return 0
            
        # 2. Expansion
        node = await self.expansion(node)
            
        # 3. Simulation
        reward = await self.simulation(node)
        
        # Update leaf node reward
        node.leaf_reward = reward
        
        # Log simulation result
        self.write_log(f"💡 [SIMULATION] Depth: {node.state['depth']}, Reward: {reward:.4f}, "
                       f"Confirmed: {node.state.get('caption', 'N/A')}, "
                       f"Missing: {node.state.get('missing_objects', [])}")

        # 4. Backpropagation
        self.backpropagation(node, reward)
        
        return reward

    async def get_final_answer(self):
        """Run MCTS to search for best answer"""
        initial_state = {
            'depth': 0,
            'image': self.image,
            'action_history': [],
            'text': self.row['question'],  # Root node uses original question as text
            'image_width': self.image_width,
            'image_height': self.image_height,
            'region_coords': (0, 0, self.image_width, self.image_height)
        }
        
        # Run multiple simulations
        self.write_log(f"\n🔍 Starting {self.n_simulations} MCTS simulations...")
        for i in range(self.n_simulations):
            self.write_log(f"\n--- Simulation {i+1}/{self.n_simulations} ---")
            await self.single_run(initial_state)
            
        # Collect all nodes
        all_nodes = []
        nodes_to_visit = [self.root]
        while nodes_to_visit:
            node = nodes_to_visit.pop()
            all_nodes.append(node)
            nodes_to_visit.extend(node.children.values())
            
        # Generate final question
        final_qs = ''
        if not is_none(self.row['hint']):
            final_qs += self.row['hint'] + '\n'
        final_qs += self.row['question']
        
        for option_char, option in zip(self.cur_option_char, self.options):
            final_qs += '\n' + option_char + '. ' + option

        if self.args.single_pred_prompt:
            if self.args.lang == 'cn':
                final_qs += '\n' + "请直接回答选项字母。"
            else:
                final_qs += '\n' + "Answer with the option's letter from the given choices directly."
            
        # Generate answer for each node
        answers = []
        for node in all_nodes:
            answer = await self.generate(final_qs, node.state['image'])
            
            # Extract option letter from answer
            for letter in ['A', 'B', 'C', 'D']:
                if letter in answer:
                    answers.append((letter, node.leaf_reward))  # Use leaf reward as weight
                    break
            else:
                answers.append(('A', node.leaf_reward))  # If no valid option found, default to A with leaf reward
                
        # Find node with highest value/visits
        best_node = max(all_nodes, key=lambda x: (x.leaf_reward, all_nodes.index(x)))
        
        self.write_log(f"\n📊 [TREE SUMMARY]")
        self.write_log(f"Total Nodes Created: {len(all_nodes)}")
        self.write_log(f"Best Node - Reward: {best_node.leaf_reward:.4f}, Depth: {best_node.state['depth']}, "
                       f"Area Ratio: {best_node.valid_area_ratio:.4f}")
        self.write_log(f"Best Node Actions: {' -> '.join(best_node.state['action_history'])}")
        
        if self.use_ensemble:
            # Weighted voting for final answer
            from collections import defaultdict
            vote_result = defaultdict(float)
            for answer, weight in answers:
                vote_result[answer] += weight
                
            # Check if all weights are zero
            if all(weight == 0 for weight in vote_result.values()):
                # Regenerate answer using original image
                answer = await self.generate(final_qs, self.image)
                # Extract option letter from answer
                for letter in ['A', 'B', 'C', 'D']:
                    if letter in answer:
                        final_answer = letter
                        break
                else:
                    final_answer = 'A'  # If no valid option found, default to A
            else:
                final_answer = max(vote_result, key=vote_result.get)
            
            self.write_log(f"\n🗳️  [ENSEMBLE VOTING]")
            for ans, weight in vote_result.items():
                self.write_log(f"  {ans}: {weight:.4f}")
            self.write_log(f"  Final Answer: {final_answer}")
        else:
            # Use best_node's answer
            final_answer = max(answers, key=lambda x: x[1])[0]
            self.write_log(f"\n✅ [FINAL ANSWER] {final_answer} (from best node)")
        
        return final_answer, final_qs, answers[-1][0], best_node.state['image'], best_node, self.root

    def serialize_tree(self, node):
        """Serialize tree structure for saving to jsonl"""
        node_info = {
            "state": node.state,
            "visits": node.visits, 
            "value": node.value,
            "leaf_reward": node.leaf_reward,
            "expert_info": node.expert_info,
            "valid_area_ratio": node.valid_area_ratio,
            "region_coords": node.region_coords,
            "extra_info": node.extra_info,
            "children": {action: self.serialize_tree(child) for action, child in node.children.items()}
        }
        return node_info
    
    def _calculate_ucb_score(self, node, parent_id):
        """Calculate UCB score for a node (for serialization purposes)"""
        if node.parent is None or node.visits == 0:
            return None
        
        try:
            total_visits = sum(child.visits for child in node.parent.children.values())
            if total_visits == 0:
                return None
            
            exploit = node.value / node.visits
            explore = math.sqrt(2 * math.log(total_visits) / node.visits)
            return exploit + self.c_puct * explore
        except:
            return None
    
    def is_similar_node(self, node_record1, node_record2, iou_threshold=0.8):
        """Check if two node records are similar based on region IoU and objects
        
        Args:
            node_record1: First node record dictionary
            node_record2: Second node record dictionary
            iou_threshold: IoU threshold for considering regions as similar (default: 0.8)
            
        Returns:
            bool: True if nodes are similar, False otherwise
        """
        # Calculate IoU of region_coords
        coords1 = node_record1.get('region_coords')
        coords2 = node_record2.get('region_coords')
        
        if coords1 is None or coords2 is None:
            return False
        
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = coords1
        x2_min, y2_min, x2_max, y2_max = coords2
        
        # Calculate intersection area
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        # Check if there is intersection
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            intersection_area = 0
        else:
            intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union area
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            iou = 0
        else:
            iou = intersection_area / union_area
        
        # Check if IoU meets threshold
        if iou < iou_threshold:
            return False
        
        # Compare missing_objects
        missing1 = set(node_record1.get('missing_objects', []))
        missing2 = set(node_record2.get('missing_objects', []))
        
        if missing1 != missing2:
            return False
        
        # Compare confirmed_objects
        confirmed1 = set(node_record1.get('confirmed_objects', []))
        confirmed2 = set(node_record2.get('confirmed_objects', []))
        
        if confirmed1 != confirmed2:
            return False
        
        return True
    
    def serialize_tree_flat(self, root_node):
        """Serialize tree structure as a flat list without hierarchy
        
        Args:
            root_node: The root node of the MCTS tree
            
        Returns:
            list: A list of dictionaries, each containing node information with a unique node_id
        """
        flat_nodes = []
        node_counter = [0]  # Use list to maintain counter in nested function
        
        def traverse_and_serialize(node, parent_id=None, action_from_parent=None):
            # Assign unique ID to current node
            current_id = node_counter[0]
            node_counter[0] += 1
            
            # Create node info dictionary
            node_info = {
                # Identification
                "question_id": self.row.get('index', 'unknown'),
                "node_id": current_id,
                "parent_id": parent_id,
                "action_from_parent": action_from_parent,
                
                # Tree structure
                "depth": node.state.get('depth', 0),
                "num_children": len(node.children),
                "is_leaf": len(node.children) == 0,
                "is_root": parent_id is None,
                "path_length": len(node.state.get('action_history', [])),
                
                # MCTS statistics
                "visits": node.visits,
                "value": node.value,
                "leaf_reward": node.leaf_reward,
                "avg_value": node.value / node.visits if node.visits > 0 else 0,
                
                # UCB score (if has parent and parent has been visited)
                "ucb_score": self._calculate_ucb_score(node, parent_id) if parent_id is not None else None,
                
                # Image region information
                "valid_area_ratio": node.valid_area_ratio,
                "region_coords": node.region_coords,
                "region_width": node.region_coords[2] - node.region_coords[0],
                "region_height": node.region_coords[3] - node.region_coords[1],
                "region_area": (node.region_coords[2] - node.region_coords[0]) * (node.region_coords[3] - node.region_coords[1]),
                "image_width": node.state.get('image_width', 0),
                "image_height": node.state.get('image_height', 0),
                
                # Action and state information
                "action_history": node.state.get('action_history', []),
                "available_actions": node.untried_actions.copy() if node.untried_actions else [],
                "num_untried_actions": len(node.untried_actions) if node.untried_actions else 0,
                "is_fully_expanded": len(node.untried_actions) == 0 if node.untried_actions is not None else True,
                
                # Content information
                "text": node.state.get('text', ''),
                "caption": node.state.get('caption', ''),
                "missing_objects": node.state.get('missing_objects', []),
                "num_confirmed_objects": len(node.state.get('caption', '').split(',')) if node.state.get('caption') else 0,
                "num_missing_objects": len(node.state.get('missing_objects', [])),
                "all_objects_found": len(node.state.get('missing_objects', [])) == 0,
                
                # Expert information
                "expert_info": node.expert_info,
                "num_expert_boxes": len(node.expert_info.get('boxes', [])) if node.expert_info else 0,
                "has_expert_info": node.expert_info is not None,
                
                # Additional information
                "extra_info": node.extra_info,
                
                # Metadata
                "question_text": self.row.get('question', ''),
                "key_objects": self.key_objects if hasattr(self, 'key_objects') else [],
            }
            
            # Add to flat list
            flat_nodes.append(node_info)
            
            # Recursively process all children
            for action, child in node.children.items():
                traverse_and_serialize(child, parent_id=current_id, action_from_parent=action)
        
        # Start traversal from root
        traverse_and_serialize(root_node)
        
        return flat_nodes

    async def _process(self):
        try:
            # Extract key objects from question
            self.key_objects = await self.extract_key_objects()
            
            final_answer, prompt, full_answer, final_image, best_node, root_node = await self.get_final_answer()
            
            # Serialize tree structure for saving
            tree_info = self.serialize_tree(root_node)

            # flat_tree_info = self.serialize_tree_flat(root_node)
            
            return {
                "question_id": self.row['index'],
                "round_id": self.round_idx,
                "prompt": prompt,
                "text": final_answer,
                "options": self.options,
                "option_char": self.cur_option_char,
                "answer_id": shortuuid.uuid(),
                "model_id": self.args.model_path,
                "answer": self.row['answer'],
            }
        finally:
            # Always close log file
            self.close_log()
