class Unit {
  String activationType;
  PVector pos;
  HashMap<Unit, Edge> predecessors;
  ArrayList<Unit> successors;
  PShape shape;
  float currentNet;
  String label;
  float activationValue;
  float activationDerivative;
  ArrayList<PShape> edges;
  boolean isInput;
  boolean isOutput;
  boolean selected;
  
  // DEBUG
  int depth = 0;

  Unit(float x, float y, String activationType, String label, boolean isInput) {
    this.activationType = activationType;
    this.pos = new PVector(x, y);
    this.predecessors = new HashMap<Unit, Edge>();
    this.successors = new ArrayList<Unit>();
    this.currentNet = 0;
    this.label = label;
    this.isInput = isInput;
    this.isOutput = false;
    this.selected = false;
    this.activationDerivative = 0;
    this.activationValue = 1;
    
    if(isInput) {
      this.shape =  createShape(ELLIPSE, 0, 0, 25, 25);
      this.shape.setStrokeWeight(3);
    } else {
        this.shape = createShape(ELLIPSE, 0, 0, 100, 100);
        this.shape.setStrokeWeight(3);
    }
  }

  void linkToInput(Unit unit) {
    predecessors.put(unit, new Edge(0.0, this.pos.x, this.pos.y, unit.pos.x, unit.pos.y));
    unit.successors.add(this); 
  }

  void select(boolean selected) {
    if(selected) {
      this.shape.setStroke(color(255, 0, 0));
    } else {
      this.shape.setStroke(color(0, 0, 0));
    }
    
    this.selected = selected;
  }
  
  
  void move(float dx, float dy) {
    pos.x += dx;
    pos.y += dy;
    
    updateEdges();
    // If the unit has successors, we update their edges as well (Useful namely for when a unit is moved. This way, the previous and following edges are reshaped accordingly
    for(Unit s : successors) {
       for (Unit unit : s.predecessors.keySet()) {
         s.predecessors.get(unit).repos(unit.pos.x, unit.pos.y, s.pos.x, s.pos.y);
         s.updateEdges();
       }
    }
    
  }
  
  void updateShape() {
    
  }
  
  void changeActivationType(String activationType) {
    this.activationType = activationType;
  }

  boolean checkMouseCollision(float x, float y) {
    boolean collided = false;
    float dist = dist(x, y, pos.x, pos.y);
    
    if(isInput)
      collided = dist < 12.5 ? true : false;
    
    else
      collided = dist < 50 ? true : false;
    
    return collided;
  }
  
  float activation() {
    if(!isInput) {
      this.currentNet = 0;
  
      for (Unit unit : predecessors.keySet()) {
        if(!unit.isInput)
          this.currentNet += predecessors.get(unit).weight * unit.activation();
        else
          this.currentNet += predecessors.get(unit).weight * unit.activationValue;
      }
        
      this.activationValue = activationFunction(this.currentNet);
      this.activationDerivative = activationFunctionDerivative(this.currentNet);
    }
    return this.activationValue;
  }
  
  void printWeights() {
    print("{ ");
    int i = 0;
    for (Unit unit : predecessors.keySet()) {
      if(i == predecessors.size() - 1)
        print("b = " + predecessors.get(unit).weight + " ");
      else
        print("w"+i+" = " + predecessors.get(unit).weight + " ");
      i++;  
    }
    print("}\n");

  }
  
  void printEdgeObject() {
    print("{ ");
    int i = 1;
    for (Unit unit : predecessors.keySet()) {
        print("edge" + i + " = " + predecessors.get(unit) + " ");
      i++;  
    }
    print("}\n");

  }
  
  void printPred() {
    print("{ ");
    int i = 0;
    for (Unit unit : predecessors.keySet()) {
        print("p"+i+" = " + unit.label + " ");
      
      i++;  
    }
    print("}\n");

  }
  
  void printSuc() {
    print("{ ");
    int i = 0;
    for (Unit unit : successors) {
        print("s"+i+" = " + unit.label + " ");
      
      i++;  
    }
    print("}\n");
  }
  
  void printUnit() {
    println("###### NAME : " + this.label + "########");
    println("###### POSITION : " + this.pos.x + ", " + this.pos.y);
    printEdgeObject();
    printWeights();
    printPred();
    printSuc();
    updateEdges();
  }

  void setActivationValue(float activationValue, ArrayList<LearningExample> set) {
    this.activationValue = activationValue;
    
    if(isInput) {
      this.shape.setFill(color(map(activationValue, minValue(this.label, set), maxValue(this.label, set), 0, 255)));
    }
  }
  
  float activationFunction(float input) {
    float a = 0;
    
    switch(this.activationType) {
    case "SIGMOID":
      a = sigmoid(input);
      break;
    case "TANH":
      a = tanh(input);
      break;
    case "RELU":
      a = relu(input);
      break;
    case "LRELU":
      a = lrelu(input);
      break;
    case "ELU":
      a = elu(input);
      break;
    case "SOFTPLUS":
      a = softplus(input);
      break;
    default:
      a = 0;
      break;
    }
      
    return a;  
  }
  
  float activationFunctionDerivative(float net) {
    switch(this.activationType) {
    case "SIGMOID":
      return sigmoid(net) * (1 - sigmoid(net));
    case "TANH":
      return 1 - pow(tanh(net), 2);
    case "SOFTPLUS":
      // The sigmoid is the derivative of softplus !
      return sigmoid(net);  
    case "RELU":
      if (net < 0)
        return 0;
      else
        return 1;
    case "ELU":
      if (net < 0)
        return 1 * exp(net);
      else
        return 1;    
    case "LRELU":
      if (net < 0)
        return 0.01;
      else
        return 1;   
    default:
      return 0;
    }
  }

  
  void updateEdges() { 
    for (Unit unit : predecessors.keySet()) {
      // Refresh the params
      predecessors.get(unit).repos(unit.pos.x, unit.pos.y, pos.x, pos.y);
      
      float weight = predecessors.get(unit).weight;
      PShape currentShape = predecessors.get(unit).shape; 
      PShape feedBall = predecessors.get(unit).feedBall; 
      currentShape.resetMatrix();
      
      boolean overflow = false;
      // To prevent visual edges to grow too big, constrain this intermediary variable value
      if(abs(weight) > 5) {
        weight = weight > 0 ? 5 : -5;
        overflow = true;
      }
        
        
      currentShape.setStrokeWeight(map(abs(weight), 0, 1, 1, 15));
      // Reset the default scale
      predecessors.get(unit).feedBall.resetMatrix();
  
      // Scales to the current weight
      // Je ne sais pas pourquoi j'ai écrit cette ligne de code en dessous, c'était la source d'un énorme BUG
      feedBall.scale(map(abs(weight), 0, 1, 1, 15) + 3);
  
      if (weight < 0) {
        currentShape.setStroke(overflow == false ? color(0, 0, 255) : color(0, 0, 160));
        feedBall.setFill(overflow == false ? color(0, 0, 255) : color(0, 0, 160));
        feedBall.setStroke(overflow == false ? color(0, 0, 255) : color(0, 0, 160));
      } else if (weight > 0) {
        currentShape.setStroke(overflow == false ? color(255, 255, 0) : color(240, 179, 0));
        feedBall.setStroke(overflow == false ? color(255, 255, 0) : color(240, 179, 0));
        feedBall.setFill(overflow == false ? color(255, 255, 0) : color(240, 179, 0));
      } else if (weight == 0) {
        currentShape.setStroke(color(0));
        feedBall.setStroke(color(0));
        feedBall.setFill(color(0));
      }
    }
  }
  
  void resetWeights(String initType) {
    for (Unit unit : predecessors.keySet()) {
      Edge edge = predecessors.get(unit);
      switch(initType) {
        case "RAND":
          edge.weight = random(-1, 1);
          break;
        case "XAVIER":
          // According to a formula found online -- random(nb_IN, nb_OUT) * sqrt(......)
          edge.weight = random(predecessors.size(), 1) * sqrt(1.0 / predecessors.size());
          break;
        case "HE":
          edge.weight = random(predecessors.size(), 1) * sqrt(2.0 / predecessors.size());
          break;  
        case "ZEROS":
          edge.weight = 0.0;
          break;
        default:
          println("error");
          break;
      }
       
       predecessors.replace(unit, edge);
    }
    
    updateEdges();
  }
  
  
  float learn(float activation, ArrayList<Float> gradients, float rate, float error) {
    float deltaWeight = 1;
    if(!isInput) {
      float gradient = 1;
      // Compose all gradients
      gradients.add(this.activationDerivative);
      for(float g : gradients)
        gradient *= g;
      
      for (Unit unit : predecessors.keySet()) {
        deltaWeight = rate * gradient * error;
        
        unit.learn(unit.activationValue, gradients, selectLearningRate(unit.activationType, "step"), error);
        
        float weight = predecessors.get(unit).weight;
  
        if(abs(weight) > 40) {
           predecessors.get(unit).weight = 40;
          println("WARNING : Exploding weight : Weight update restricted. Learning will be affected.");
        } else {
           predecessors.get(unit).weight = weight + deltaWeight;
           updateEdges();
        }
      }
    }
    return deltaWeight;
  }

  void tirets(int nb) {
    nb *= 20;
    for(int i = 0; i < nb; i++)
      print(' ');
  }
  
  void printgr(ArrayList<Float> gr) {
    print("{");
     for(float g : gr)
        print(g + ", ");
    print("}");    
  }
  
  void epoch(float a, ArrayList<Float> gradients, float e, Unit previous, int d) {
    float gradient = 1;
      // Compose all gradients
      if(previous != null)
       // DA / Dnet
        gradients.add(this.activationDerivative);
      for(float g : gradients)
        gradient *= g;
    
    //tirets(d);
    //println("UNIT : " + label);
    
    //tirets(d);
    //print("Gradients présents : ");
    //printgr(gradients);
    //println("A ce niveau, la valeur des gradients cumulés est " + gradient);
    
    if(predecessors.keySet().size() > 0) {
      for(Unit unit : predecessors.keySet()) {
        // * Dnet / Dan
        predecessors.get(unit).deltaWeight += gradient * unit.activationValue;     

        unit.epoch(activationValue, (ArrayList<Float>) gradients.clone(), e, this, d + 1);
        unit.finishEpoch(e, d + 1);
        
        if(new Float(predecessors.get(unit).deltaWeight).isNaN()) {
          //println("Error, could not compute activation and gradient, net too high, or gradient activationValue is exploding. Try a lower learning rate.");
          predecessors.get(unit).deltaWeight = 0;
          //println(unit.label + " was NaN"); 
        }
      }
    } else {
      //println("null keyset");
    }    
  }

  void finishEpoch(float e, int d) {
    // Finally makes the individual units learn
    for (Unit unit : predecessors.keySet()) {
      //tirets(d);
      //println("For " + unit.label + ", before finishing, deltaWeight is " + predecessors.get(unit).deltaWeight);
      float newWeight = predecessors.get(unit).weight - 0.25 * predecessors.get(unit).deltaWeight * e;
      if(new Float(newWeight).isNaN()) {
        println("weight before NaN : " + predecessors.get(unit).weight);
        return;
      }
      predecessors.get(unit).weight -= 2 * predecessors.get(unit).deltaWeight * e;
      predecessors.get(unit).deltaWeight = 0;
    }
 
  // tirets(d);
   //println("END EPOCH FOR : " + label);
    
  }

  void animate(float time) {
    this.shape.setStroke(color(0, 0, 0));

    for (Unit unit : predecessors.keySet()) {
      shape(predecessors.get(unit).feedBall, interpolate(unit.pos.x, this.pos.x, time, "SIGMOID"), interpolate(unit.pos.y, this.pos.y, time, "SIGMOID"));
    }
  }

  void drawObject() {
    pushStyle();
    pushMatrix();
    fill(255);

    // Draw the edges linking it to each input associated
    for(Unit unit : predecessors.keySet()) {
      unit.drawObject();
      PShape shape = predecessors.get(unit).shape;
      shape(shape);
      
    }

    if(isInput) {    
      pushStyle();
      pushMatrix();
        fill(0);
        text(this.label, this.pos.x - 40, this.pos.y + 8);
      popMatrix();  
      popStyle();
    }
   
    shape(this.shape, this.pos.x, this.pos.y);
    popMatrix();  
    popStyle();
  }
  
  void drawActivationFunction(float posX, float posY, float minX, float maxX, float w, float h, float[] outactivationValues, color[] cols) {
    pushMatrix();
    pushStyle();
    strokeWeight(2);
    translate(posX - w / 2, posY + h / 2);
    beginShape();
    for (int i = 0; i < w; i += 2) {
      float x = map(i, 0, w, minX, maxX);
      switch(this.activationType) {
      case "SIGMOID":
        curveVertex(i, -h * sigmoid(x));
        break;
      case "TANH":
        //curveVertex(i, -h * tanh(x));
        curveVertex(i, -h * (tanh(x) + 1) / 2);
        break;
      case "SOFTPLUS":
        //curveVertex(i, -h * tanh(x));
        //if(-h * softplus(x) >= -h)
          curveVertex(i, map(softplus(x), 0, maxX, 0, -h));
          //curveVertex(i, -h * softplus(x));
        break;  
      case "RELU":
        curveVertex(i, map(relu(x), 0, maxX, 0, -h) -h / 4);
        break;
      case "LRELU":
        curveVertex(i, map(lrelu(x), 0, maxX, 0, -h) -h / 4);
        break;  
      
      case "ELU":
        curveVertex(i, map(elu(x), 0, maxX, 0, -h) -h / 4);
        break;  
      }
    }
    endShape();

    if (outactivationValues != null && cols != null && outactivationValues.length == cols.length) {
      //println("minX : " + minX + "maxX : " + maxX + " out : " + outactivationValues[0] + " target :" + outactivationValues[1]);
      noStroke();
      for (int i = 0; i < outactivationValues.length; i++) {
        fill(cols[i]);
        float ex = 100;
        switch(activationType) {
        case "SIGMOID":
          ex = map(logit(outactivationValues[i]), minX, maxX, 0, w);
          break;
        case "TANH":
          ex = map(argtanh(outactivationValues[i]), minX, maxX, 0, w);
          break;
        case "SOFTPLUS":
          ex = map(log(exp(outactivationValues[i]) - 1), minX, maxX, 0, w);
          break;
        case "RELU":
          ex = map(outactivationValues[i], minX, maxX, 0, w);  
          break;
        case "LRELU":
          ex = map(outactivationValues[i], minX, maxX, 0, w);  
          break;
        case "ELU":
          ex = map(elu_inverse(outactivationValues[i]), minX, maxX, 0, w);  
          break;    
          
        default:
          println("error");
        }
        // println("net = " + this.currentNet + " ex = "+ ex);
        float ey = 0;
        if (this.activationType == "RELU" || this.activationType == "LRELU" || this.activationType == "ELU")
          ey = map(outactivationValues[i], 0, maxX, 0, -h) -h / 4; 
        else if (this.activationType == "TANH") {
          ey = -h * (outactivationValues[i] + 1) / 2;
        } else if (this.activationType == "SOFTPLUS") {
          ey = map(outactivationValues[i], 0, maxX, 0, -h);
        } else
          ey = -h * outactivationValues[i];

        ellipse(ex, ey, 8, 8);
      }
    }

    popMatrix();
    popStyle();
  }
  
}
