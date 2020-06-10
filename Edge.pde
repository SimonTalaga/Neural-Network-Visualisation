class Edge {
  float weight;
  // This variable can be accumulated as a "charge" that can be discharged to update the weight
  float deltaWeight;
  // The visual aspect of the link between 2 units.
  PShape shape;
  PShape feedBall;

  Edge(float w, float fromX, float fromY, float toX, float toY) {
    this.weight = w;
    this.shape = createShape(LINE, fromX, fromY, toX, toY);
    this.deltaWeight = 0;
    this.feedBall = createShape(ELLIPSE, 0, 0, 1, 1);
  }
  
  void repos(float fromX, float fromY, float toX, float toY) {
    this.shape = createShape(LINE, fromX, fromY, toX, toY);
  }
}
