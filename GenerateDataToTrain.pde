class LearningExample {
  float x;
  float y;
  float target;

  LearningExample(float x, float y, float target) {
    this.x = x;
    this.y = y;
    this.target = target;
  }
}

ArrayList<LearningExample> points = new ArrayList<LearningExample>();
float targetValue = 0.5;

void setup() {
  size(750, 750);
  background(255);
}


void draw() {
  background(255);
  fill(map(targetValue, 0, 1, 0, 255));
  rect(10, 10, 50, 50);
  
  for(LearningExample point : points) {
    fill(map(point.target, 0, 1, 0, 255));
    ellipse(map(point.x, -1, 1, 0, width), map(point.y, -1, 1, 0, height), 15, 15);
  }
}



void mouseReleased() {
  points.add(new LearningExample(map(mouseX, 0, width, -1, 1), map(mouseY, 0, height, -1, 1), targetValue));
}

void mouseWheel(MouseEvent event) {
  float e = event.getCount();
  targetValue += 0.05 * e;
  if(targetValue > 1)
    targetValue = 1;
  else if(targetValue < 0)
    targetValue = 0;  
}

void keyPressed() {
  if (key == 'p') {
    print("{ ");
    for(LearningExample point : points) {
      print(point.x + " " + point.y + " " + point.target + " ");
    }
    print(" }");
  } else if (key == 's') {
    // Prints the current plot in an image
   saveFrame("plots/graph.png");
  } else if (key == 'f') {
    points = new ArrayList<LearningExample>();
    
  }
}
