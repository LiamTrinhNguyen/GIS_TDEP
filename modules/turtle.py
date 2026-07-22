from js import document, window
import math

class EtchTurtle:
    def __init__(self, canvas_id="canvas"):
        self.canvas = document.getElementById(canvas_id)
        self.ctx = self.canvas.getContext("2d")
        self.width = self.canvas.width
        self.height = self.canvas.height
        
        # Starting position (center)
        self.x = self.width / 2
        self.y = self.height / 2
        self.angle = 0          # degrees (0 = right)
        self.pen_down = True
        self.color = "#00FF41"
        self.line_width = 2
        
        self.clear()

    def clear(self):
        self.ctx.fillStyle = "#000C00"
        self.ctx.fillRect(0, 0, self.width, self.height)
        self.x = self.width / 2
        self.y = self.height / 2
        self.angle = 0
        self.pen_down = True

    def forward(self, distance):
        rad = math.radians(self.angle)
        new_x = self.x + distance * math.cos(rad)
        new_y = self.y + distance * math.sin(rad)

        if self.pen_down:
            self.ctx.strokeStyle = self.color
            self.ctx.lineWidth = self.line_width
            self.ctx.beginPath()
            self.ctx.moveTo(self.x, self.y)
            self.ctx.lineTo(new_x, new_y)
            self.ctx.stroke()

        self.x = new_x
        self.y = new_y

    def turn(self, degrees):
        self.angle = (self.angle + degrees) % 360

    def left(self, degrees):
        self.turn(-degrees)

    def right(self, degrees):
        self.turn(degrees)

    def penup(self):
        self.pen_down = False

    def pendown(self):
        self.pen_down = True

    def center(self):
        self.x = self.width / 2
        self.y = self.height / 2
        self.angle = 0

    def goto(self, x, y):
        if self.pen_down:
            self.ctx.beginPath()
            self.ctx.moveTo(self.x, self.y)
            self.ctx.lineTo(x, y)
            self.ctx.stroke()
        self.x = x
        self.y = y