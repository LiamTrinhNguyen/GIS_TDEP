from modules.turtle import EtchTurtle

# Global turtle instance (created once)
turtle = None

def init_turtle():
    global turtle
    if turtle is None:
        turtle = EtchTurtle("canvas")
    return turtle

def run_pipsqueak(code: str):
    t = init_turtle()
    
    lines = [line.strip() for line in code.strip().splitlines() if line.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if line.startswith("#"):
            i += 1
            continue
            
        # REPEAT n { ... }
        if line.upper().startswith("REPEAT"):
            try:
                parts = line.upper().replace("{", " {").split()
                count = int(parts[1])
                # Find matching }
                block = []
                i += 1
                depth = 1
                while i < len(lines) and depth > 0:
                    if "{" in lines[i]:
                        depth += 1
                    if "}" in lines[i]:
                        depth -= 1
                    if depth > 0:
                        block.append(lines[i])
                    i += 1
                
                for _ in range(count):
                    run_pipsqueak("\n".join(block))
                continue
            except Exception as e:
                print(f"REPEAT error: {e}")
        
        # Simple commands
        tokens = line.upper().split()
        cmd = tokens[0] if tokens else ""
        
        try:
            if cmd == "FORWARD" or cmd == "FD":
                t.forward(float(tokens[1]))
            elif cmd == "BACK" or cmd == "BK":
                t.forward(-float(tokens[1]))
            elif cmd == "TURN" or cmd == "RIGHT" or cmd == "RT":
                t.right(float(tokens[1]))
            elif cmd == "LEFT" or cmd == "LT":
                t.left(float(tokens[1]))
            elif cmd == "CENTER":
                t.center()
            elif cmd == "PENUP" or cmd == "PU":
                t.penup()
            elif cmd == "PENDOWN" or cmd == "PD":
                t.pendown()
            elif cmd == "CLEAR":
                t.clear()
            elif cmd == "GOTO":
                t.goto(float(tokens[1]), float(tokens[2]))
            else:
                print(f"Unknown command: {line}")
        except Exception as e:
            print(f"Error on '{line}': {e}")
        
        i += 1