// OBSERVATION : Contradiction entre la couleur des points affichés par rapport à l'activation au fond et la courbe d'apprentissage qui dit que l'erreur est minimisée << C'est que la distribution est aléatoire !
// Revoir la méthode d'apprentissage : c'est peut-être pas exactement activationDerivative(a) <<< effectivement c'était pété.

// BUG ACTUNIT2 ne s'update jamais.
//>>> Le problème vient des linktoinput
// C'est la liste des gradients bordel, c'est le même objet tout du long, il faut la flush

// GOOOOOOOOOOOD

// Le training set est volontairement bruité
// 04/04 Ajout de la normalisation des données

float selectLearningRate(String type, String mode) {
  switch(type) {
      case "SIGMOID":
        if(mode == "step")
          return 0.05;
        else if(mode == "batch")
          return 1.2;
        else
          break;

      case "TANH":
        if(mode == "step")
          return 0.5;
        else if(mode == "batch")
          return 0.05;
        else
          break;

      case "RELU":
        if(mode == "step")
          return 0.003;
        else if(mode == "batch")
          return 0.003;
        else
          break;
          
      case "LRELU":
        if(mode == "step")
          return 0.3;
        else if(mode == "batch")
          return 0.0008;
        else
          break;    
      
      case "ELU":
        if(mode == "step")
          return 0.3;
        else if(mode == "batch")
          return 0.0005;
        else
          break;    
              
      case "SOFTPLUS":
        if(mode == "step")
          return 1;
        else if(mode == "batch")
          return 0.1;
        else
          break;        
      }
  
  return 1;      
}


void scaleNetworkWeights(float f) {
  for(ArrayList<Unit> layer : layers) {
      for(Unit unit : layer) {
        for (Unit punit : unit.predecessors.keySet()) {
          unit.predecessors.get(punit).weight /= f;
        }
        
        unit.updateEdges();
      }
  }
}


// Variables used for GUI
float previousMouseClickX, previousMouseClickY;


//// Raw data from the coursework 2 
//String trainingRaw = "-1.4159 2.8613 0.7440 -4.5071 -3.0433 0.5410 -4.8815 1.1076 0.6872 3.1028 4.1846 0.7611 -4.7350 0.3086 0.4932 -0.3209 -2.5102 0.5814 4.1818 3.1635 0.7281 2.2735 -1.4237 0.5130 4.9341 -2.7584 0.5708 3.4159 -3.6783 0.4821 1.0251 -4.1299 0.5128 -1.8929 3.1622 0.5052 -1.7427 2.4389 0.6184 -4.0696 4.1184 0.4816 -2.7650 -2.8310 0.7648 2.3032 -0.5474 0.7583 -4.3810 1.9013 0.6493 2.7801 -4.5528 0.7458 -1.1568 -2.0832 0.7581 3.1837 2.0726 0.5960";
//String validationRaw = "2.2972 1.9126 0.6027 -0.3938 1.3860 0.5371 3.3180 -4.0483 0.7664 -1.2741 1.1743 0.5232 0.2497 4.9123 0.4823 -0.0729 2.5009 0.5098 -1.4399 2.5700 0.4919 4.3122 3.2504 0.6061 4.3671 0.1367 0.7403 -1.1906 3.5367 0.4844 -4.6318 -4.9067 0.7063 1.5573 -0.2311 0.6913 -0.3303 -2.7632 0.7334 -3.3171 -3.2955 0.6766 4.3487 0.4051 0.7342 0.7068 -4.1512 0.7609 -3.1608 2.9360 0.4803 2.0517 -1.2512 0.7365 0.2966 -4.5806 0.7617 -1.5150 -2.0103 0.6750";
//String evaluationRaw = "1.1043 0.3999 0.6423 0.0589 0.6830 0.5847 -3.5480 -1.4366 0.5659 4.9814 -0.5973 0.7563 4.8947 1.8805 0.6981 -0.2027 3.8749 0.4873 -1.9887 1.9777 0.4946 3.7548 -1.1792 0.7539 0.7016 1.7527 0.5536 3.9330 -3.3644 0.7660 3.9028 1.1419 0.7009 -4.3450 -1.8746 0.5609 -1.6200 -4.3106 0.7461 4.6483 -3.3262 0.7667 -4.5126 -2.7378 0.6003 -1.7045 -4.9982 0.7542 -3.2534 -4.2470 0.7173 1.4696 -2.2162 0.7471 1.7505 -2.4501 0.7528 4.5582 1.0843 0.7192";

//// RAW DATA FROM CUSTOM SOFTWARE
String trainingRaw = "-0.59466666 -0.584 1.0 -0.39733332 -0.816 1.0 -0.04000002 -0.8986667 1.0 0.3573333 -0.85333335 1.0 0.576 -0.61333334 1.0 0.7066667 -0.22666669 1.0 0.72 0.09866667 1.0 0.61333334 0.32799995 1.0 0.40533328 0.5866667 1.0 -0.055999994 0.6666666 1.0 -0.51199996 0.56533337 1.0 -0.67733335 0.15999997 1.0 -0.74666667 -0.22933334 1.0 -0.28266668 -0.22666669 0.5999999 -0.16266668 -0.36799997 0.5999999 0.16266668 -0.39466667 0.5999999 0.31733334 -0.14933336 0.5999999 0.288 0.1146667 0.5999999 0.106666684 0.26133335 0.5999999 -0.26133335 0.24266672 0.5999999 -0.384 0.018666625 0.5999999 -0.33866668 -0.12533331 0.5999999 -0.19466668 -0.096000016 0.19999984 -0.023999989 -0.20533335 0.19999984 0.14666665 -0.144 0.19999984 0.13600004 0.05066669 0.19999984 0.026666641 0.17866671 0.19999984 -0.15200001 0.069333315 0.19999984 0.03999996 -0.06133336 0.19999984 -0.042666674 -0.106666684 0.19999984 -0.010666668 0.0053333044 0.19999984 -0.010666668 -0.05066669 0.0 -0.9546667 -0.79466665 0.7000001 -0.7626667 -0.992 0.7000001 0.6533333 -0.968 0.7000001 0.896 -0.78933334 0.7000001 -0.856 0.7786666 0.7000001 -0.736 0.92266667 0.7000001 0.928 0.74666667 0.7000001 0.76 0.9306667 0.7000001 -0.792 -0.7066667 1.0 -0.90933335 -0.36266667 1.0 -0.88266665 0.20533335 1.0 -0.792 0.60800004 1.0 -0.416 0.86399996 1.0 0.26133335 0.85066664 1.0 0.7626667 0.584 1.0 0.80799997 -0.45599997 1.0";
String validationRaw = "-0.59466666 -0.584 1.0 -0.39733332 -0.816 1.0 -0.04000002 -0.8986667 1.0 0.3573333 -0.85333335 1.0 0.576 -0.61333334 1.0 0.7066667 -0.22666669 1.0 0.72 0.09866667 1.0 0.61333334 0.32799995 1.0 0.40533328 0.5866667 1.0 -0.055999994 0.6666666 1.0 -0.51199996 0.56533337 1.0 -0.67733335 0.15999997 1.0 -0.74666667 -0.22933334 1.0 -0.28266668 -0.22666669 0.5999999 -0.16266668 -0.36799997 0.5999999 0.16266668 -0.39466667 0.5999999 0.31733334 -0.14933336 0.5999999 0.288 0.1146667 0.5999999 0.106666684 0.26133335 0.5999999 -0.26133335 0.24266672 0.5999999 -0.384 0.018666625 0.5999999 -0.33866668 -0.12533331 0.5999999 -0.19466668 -0.096000016 0.19999984 -0.023999989 -0.20533335 0.19999984 0.14666665 -0.144 0.19999984 0.13600004 0.05066669 0.19999984 0.026666641 0.17866671 0.19999984 -0.15200001 0.069333315 0.19999984 0.03999996 -0.06133336 0.19999984 -0.042666674 -0.106666684 0.19999984 -0.010666668 0.0053333044 0.19999984 -0.010666668 -0.05066669 0.0 -0.9546667 -0.79466665 0.7000001 -0.7626667 -0.992 0.7000001 0.6533333 -0.968 0.7000001 0.896 -0.78933334 0.7000001 -0.856 0.7786666 0.7000001 -0.736 0.92266667 0.7000001 0.928 0.74666667 0.7000001 0.76 0.9306667 0.7000001 -0.792 -0.7066667 1.0 -0.90933335 -0.36266667 1.0 -0.88266665 0.20533335 1.0 -0.792 0.60800004 1.0 -0.416 0.86399996 1.0 0.26133335 0.85066664 1.0 0.7626667 0.584 1.0 0.80799997 -0.45599997 1.0";
String evaluationRaw = "-0.59466666 -0.584 1.0 -0.39733332 -0.816 1.0 -0.04000002 -0.8986667 1.0 0.3573333 -0.85333335 1.0 0.576 -0.61333334 1.0 0.7066667 -0.22666669 1.0 0.72 0.09866667 1.0 0.61333334 0.32799995 1.0 0.40533328 0.5866667 1.0 -0.055999994 0.6666666 1.0 -0.51199996 0.56533337 1.0 -0.67733335 0.15999997 1.0 -0.74666667 -0.22933334 1.0 -0.28266668 -0.22666669 0.5999999 -0.16266668 -0.36799997 0.5999999 0.16266668 -0.39466667 0.5999999 0.31733334 -0.14933336 0.5999999 0.288 0.1146667 0.5999999 0.106666684 0.26133335 0.5999999 -0.26133335 0.24266672 0.5999999 -0.384 0.018666625 0.5999999 -0.33866668 -0.12533331 0.5999999 -0.19466668 -0.096000016 0.19999984 -0.023999989 -0.20533335 0.19999984 0.14666665 -0.144 0.19999984 0.13600004 0.05066669 0.19999984 0.026666641 0.17866671 0.19999984 -0.15200001 0.069333315 0.19999984 0.03999996 -0.06133336 0.19999984 -0.042666674 -0.106666684 0.19999984 -0.010666668 0.0053333044 0.19999984 -0.010666668 -0.05066669 0.0 -0.9546667 -0.79466665 0.7000001 -0.7626667 -0.992 0.7000001 0.6533333 -0.968 0.7000001 0.896 -0.78933334 0.7000001 -0.856 0.7786666 0.7000001 -0.736 0.92266667 0.7000001 0.928 0.74666667 0.7000001 0.76 0.9306667 0.7000001 -0.792 -0.7066667 1.0 -0.90933335 -0.36266667 1.0 -0.88266665 0.20533335 1.0 -0.792 0.60800004 1.0 -0.416 0.86399996 1.0 0.26133335 0.85066664 1.0 0.7626667 0.584 1.0 0.80799997 -0.45599997 1.0";


//// Raw data from the coursework 1 (TEST ONLY)
//String trainingRaw = "0.6476 -4.5220 0.0307 0.9455 2.5865 0.7930 -4.9833 4.6108 0.9982 -1.8146 -4.6089 0.1153 3.2482 4.1056 0.7359 -1.3858 -0.9771 0.5616 -0.4863 3.0578 0.9264 -1.5640 0.4851 0.7987 -4.0313 0.4007 0.9426 -0.0753 2.7784 0.8899 1.4038 1.4924 0.5750 2.5206 4.8354 0.8779 -4.1061 -1.9227 0.7717 -4.7461 -0.6084 0.9257 -3.0392 1.9606 0.9643 1.6574 2.0751 0.6360 0.6059 -3.8808 0.0483 -2.7569 -0.3598 0.8179 1.8513 -4.8520 0.0120 -2.0093 -2.8508 0.3340";
//String validationRaw = "-1.1455 3.1465 0.9521 -3.9233 -1.2363 0.8304 3.4143 0.4860 0.1668 1.1522 2.3831 0.7459 4.9475 -1.3930 0.0210 3.6254 -0.2349 0.0962 -1.6330 -2.1375 0.3974 -0.4497 -0.1916 0.5586 -2.5371 3.6065 0.9844 1.6675 -2.4809 0.0668 4.1351 -3.1476 0.0101 0.6857 -2.3224 0.1260 2.5686 -0.8861 0.1129 -2.5842 0.7216 0.8962 0.5578 -1.7284 0.1908 -3.0797 -0.0270 0.8731 -3.2978 0.3599 0.9114 -2.3516 -4.8303 0.1335 0.1946 -4.0870 0.0533 0.2470 -4.1572 0.0493";
//String evaluationRaw = "2.2700 0.0293 0.2242 -4.6287 2.8322 0.9923 -3.2143 1.4797 0.9554 3.9552 -2.7481 0.0148 3.4113 3.1829 0.5698 -3.8642 3.2435 0.9909 3.2724 -3.0668 0.0178 -0.6801 0.7535 0.7380 -1.3900 0.3400 0.7635 -4.0348 4.7624 0.9971 -1.1259 3.0496 0.9484 -2.6073 0.9353 0.9104 -2.4172 4.1921 0.9888 -1.2878 4.5166 0.9826 -3.6759 -4.2716 0.3352 1.5303 2.7013 0.7451 0.9667 4.2462 0.9236 -1.1677 3.3231 0.9580 4.1678 -4.6469 0.0035 -1.2736 -4.5142 0.0915";

ArrayList<LearningExample> training, validation, evaluation;
ArrayList<LearningExample>[] activeSets;

String[] availableTypes = {"SIGMOID", "TANH", "RELU", "LRELU", "SOFTPLUS", "ELU"};
String[] availableGraphs = {"data", "learningCurve"};
String[] availableInits = {"XAVIER", "HE", "RAND", "ZEROS"};
color[] learningCols = {color(255, 0, 0), color(0, 20, 255), color(0, 255, 25) };

int typeIterator = 100;
int graphIterator = 100;
int initsIterator = 100;

String graphToDraw = "data";

// Objets

ArrayList<ArrayList<Unit>> layers;

Unit actUnit, actUnit2, actUnit3, actUnit4, actUnit5, actUnit6;
Unit inputUnitX, inputUnitY, inputBiais;

// A time variable used for interpolation
float t = 0;
int nbEpochs;
float dataScaleFactor = 0.05;

ArrayList<PVector[]> dataPoints = null;

// ----- DYNAMIC SHAPES
// GUI
String targetValue = null;
String errorValue = null;
String activationValue = null;

boolean stepLearning = false;
boolean presentExample = false;
boolean batchLearning = false;

int toLearnIndex = 0;
int currentLayerIndex = 1; // 0 is the input layer

// The factor attribute is used to scale the data. if it equals 0, then the data will be leaved unchanged, and if it equals 1, it will be normalized.
ArrayList<LearningExample> makeLearningData(String raw, float factor) {
  String[] splitData = split(raw, " ");
  ArrayList<LearningExample> data = new ArrayList<LearningExample>();

  for (int i = 0; i < splitData.length; i++) {
    if (i % 3 == 0) {
      float x = float(splitData[i]);
      float y = float(splitData[i + 1]);
      float t = float(splitData[i + 2]);
      
      x *= factor;
      y *= factor;
      t *= factor;
  
      data.add(new LearningExample(x, y, t));
    }
  }
  return data;
}

ArrayList<LearningExample> normalizeLearningData(ArrayList<LearningExample> data) {
  
  ArrayList<LearningExample> newData = new ArrayList<LearningExample>();

  for (LearningExample example : data) {      
    float x = (example.x - minValue("x", data)) / (maxValue("x", data) - minValue("x", data));
    float y = (example.y - minValue("y", data)) / (maxValue("y", data) - minValue("y", data));
    float t = (example.target - minValue("t", data)) / (maxValue("t", data) - minValue("t", data));
    newData.add(new LearningExample(x, y, t));
  }

  return newData;
}


void setup() {
  size(1500, 750);
  background(255);

  nbEpochs = 10000;

  PFont font = loadFont("SourceSansPro-Regular-48.vlw");
  textFont(font);
  

  training = normalizeLearningData(makeLearningData(trainingRaw, 1));
  validation = normalizeLearningData(makeLearningData(validationRaw, 1));
  evaluation = normalizeLearningData(makeLearningData(evaluationRaw, 1));
  
  //training = makeLearningData(trainingRaw, 1);
  //validation = makeLearningData(validationRaw, 1);
  //evaluation = makeLearningData(evaluationRaw, 1);
  
  activeSets = new ArrayList[3];
  activeSets[0] = training;
  activeSets[1] = validation; 
  activeSets[2] = evaluation;
  

  layers = new ArrayList<ArrayList<Unit>>();
  ArrayList<Unit> layer1 = new ArrayList<Unit>();
  ArrayList<Unit> layer2 = new ArrayList<Unit>();
  ArrayList<Unit> layer3 = new ArrayList<Unit>();
  ArrayList<Unit> layer4 = new ArrayList<Unit>();
  
  // Units network design
  actUnit = new Unit(209, 173, "SIGMOID", "act", false);
  actUnit2 = new Unit(209, 310, "SIGMOID", "act2", false);
  actUnit3 = new Unit(209, 451, "SIGMOID", "act3", false);
  actUnit4 = new Unit(365, 250, "SIGMOID", "act4", false);
  actUnit5 = new Unit(365, 385, "SIGMOID", "act5", false);
  actUnit6 = new Unit(525, 321, "SIGMOID", "act6", false);
  
  inputUnitX = new Unit(50, 150, "LINEAR", "X", true);
  inputUnitY = new Unit(50, 250, "LINEAR", "Y", true);
  inputBiais = new Unit(50, 350, "LINEAR", "1", true);
  Unit biais2 = new Unit(208, 562, "LINEAR", "1", true);
  Unit biais3 = new Unit(363, 531, "LINEAR", "1", true);
  
  actUnit.linkToInput(inputUnitX);
  actUnit.linkToInput(inputUnitY);
  actUnit.linkToInput(inputBiais);
  
  actUnit2.linkToInput(inputUnitX);
  actUnit2.linkToInput(inputUnitY);
  actUnit2.linkToInput(inputBiais);
  
  actUnit3.linkToInput(inputUnitX);
  actUnit3.linkToInput(inputUnitY);
  actUnit3.linkToInput(inputBiais);
  
  actUnit4.linkToInput(actUnit);
  actUnit4.linkToInput(actUnit2);
  actUnit4.linkToInput(actUnit3);
  actUnit4.linkToInput(biais2);
  
  actUnit5.linkToInput(actUnit);
  actUnit5.linkToInput(actUnit2);
  actUnit5.linkToInput(actUnit3);
  actUnit5.linkToInput(biais2);
  
  actUnit6.linkToInput(actUnit4);
  actUnit6.linkToInput(actUnit5);
  actUnit3.linkToInput(biais3);
  
  // This unit is the output unit
  actUnit6.isOutput = true;
  
  // Configure the layers 
  layer1.add(inputUnitX);
  layer1.add(inputUnitY);
  layer1.add(inputBiais);
  
  layer2.add(actUnit);
  layer2.add(actUnit2);
  layer3.add(actUnit3);
  layer2.add(biais2);
  
  layer3.add(actUnit4);
  layer3.add(actUnit5);
  layer3.add(biais3);
  
  layer4.add(actUnit6);
  
  layers.add(layer1);
  layers.add(layer2);
  layers.add(layer3);
  layers.add(layer4);

  // Initialize the weights with the current initialization method
  //actUnit.resetWeights(availableInits[abs(initsIterator) % availableInits.length]);
  
  drawGraph();
}

Unit findOutputUnit() {
  for(ArrayList<Unit> layer : layers)
    for(Unit unit : layer)
      if(!unit.isInput && unit.isOutput)
        return unit;
  
  return null;
}

Unit findSelectedUnit() {
  for(ArrayList<Unit> layer : layers)
    for(Unit unit : layer)
      if(unit.selected)
        return unit;
  
  return null;
}

float maxValue(String param, ArrayList<LearningExample> data) {
  float currentMax = 0;

  for (int i = 0; i < data.size(); i++) {
    switch(param) {
    case "x":
      if (data.get(i).x > currentMax || i == 0)
        currentMax = data.get(i).x;
      break;
    case "y":
      if (data.get(i).y > currentMax || i == 0)
        currentMax = data.get(i).y;
      break;
    case "t":
      if (data.get(i).target > currentMax || i == 0)
        currentMax = data.get(i).target;
      break;  
    default:
      currentMax = 1;
      break;
    }
  }

  return currentMax;
}


float computeError(float activation, float target) {
  return target - activation;
}


float minValue(String param, ArrayList<LearningExample> data) {
  float currentMin = 0;

  for (int i = 0; i < data.size(); i++) {
    switch(param) {
    case "x":
      if (data.get(i).x < currentMin || i == 0)
        currentMin = data.get(i).x;
      break;
    case "y":
      if (data.get(i).y < currentMin || i == 0)
        currentMin = data.get(i).y;
      break;
    case "t":
      if (data.get(i).target < currentMin || i == 0)
        currentMin = data.get(i).target;
      break;  
    default:
      currentMin = 0;
      break;
    }
  }

  return currentMin;
}

float sigmoid(float x) {
  return 1 / (1 + exp(-x));
}

float tanh(float x) {
  return (exp(2 * x) - 1) / (exp(2 * x) + 1);
}

float argtanh(float x) {
  return 1/2.0 * (log(1 + x) - log(1 - x));
}

float relu(float x) {
  return max(0, x);
}

float elu(float x) {
  if(x > 0)
    return x;
  else
    return 1 * (exp(x) - 1);
}

float lrelu(float x) {
  return max(0.01 * x, x);
}

float softplus(float x) {
  return log(1 + exp(x));
}

// Fonction réciproque de sigmoide
float logit(float x) {
  return log(x / (1 - x));
}

// Fonction réciproque de elu
float elu_inverse(float x) {
  if(x > 0)
    return x;
  else
    return 1 * log(x + 1);
}

float dsig(float x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

float interpolate(float from, float to, float t, String type) {
  if (t > 1)
    t = 1;

  // Time evolves linarly
  float formula = t;

  if (type == "SIGMOID") {
    float interval = 5;
    t = (t * 2 * interval) - interval;
    formula = sigmoid(1.5 * t);
  }
  return formula * to + (1 - formula) * from;
} 

// float minValue : the min value to be represented on the scale (= error level for iteration 0), nbEpochs the maximum
// float logBase : The number whose multiples we want to appear on the scale. 
// float nbGrad : The number of graduations inside the steps
void drawGrid(float axisSize, String direction, float minValue, float maxValue, float logBase, int nbGrads, float captionTextSize) {
  pushMatrix();
  pushStyle();
    strokeWeight(1);
    float n = maxValue / minValue;
    // How many multiplications by base are necessary to reach n ?
    float nbStep = logn(n, logBase);
    float stepPos = 0;
    
    if(direction == "x") {
      for(float step = 0; step <= nbStep; step++) {
          stepPos = step * (axisSize * log(logBase) / log(n));
          strokeWeight(1);
          stroke(240);
          for(int grad = 0; grad < nbGrads; grad++) {
             float gradPos = stepPos + axisSize * log(grad) / log(n);
             line(gradPos, 0, gradPos, axisSize);    
          }
          
          stroke(200);
          fill(200);
          textSize(captionTextSize);
          String caption = str(int(0.1 * pow(logBase, step)));
          line(stepPos, 0, stepPos, axisSize);
          text(caption, stepPos - textWidth(caption) - 5, captionTextSize);
        }      
    }
    
    else if(direction == "y") {
      for(float step = nbStep; step >= 0; step--) {
        stepPos = axisSize - step * (axisSize * log(logBase) / log(n));
        strokeWeight(1);
        stroke(240);
        for(int grad = nbGrads - 1; grad >= 0; grad--) {
           float gradPos = stepPos - (axisSize * log(grad) / log(n));
           line(0, gradPos, axisSize, gradPos);  
        }
        
        stroke(200);
        fill(200);
        
        line(0, stepPos, axisSize, stepPos);
        
        textSize(captionTextSize);
        String caption = str(int(0.1 * pow(logBase, step)));
        text(caption, 5, stepPos + captionTextSize);
        //println(step + " + " + stepPos);
      }    
    }

    popMatrix();
    popStyle(); 
}

void learningGraph(color[] colors) {
  // Draw the graph lines
  if (dataPoints != null) {
    pushMatrix();
    pushStyle();
      strokeWeight(1);
      translate(width / 2, 0);
      
      drawGrid(width / 2, "x", 0.1, nbEpochs, 10, 10, 14);
      drawGrid(height, "y", 0.1, 1000, 10, 10, 14);
      
      strokeWeight(2);
      noFill();
      for(int i = 0; i < dataPoints.size(); i++) {
        stroke(colors[i]);
        beginShape();
          for (int j = 0; j < dataPoints.get(i).length; j++) {
            vertex(dataPoints.get(i)[j].x, dataPoints.get(i)[j].y);
          }
        endShape();
      }
    popMatrix();
    popStyle();   
  }
}

void dataGraph(ArrayList<LearningExample>[] examples, color[] colors) {
  pushMatrix();
  pushStyle();
  translate(width / 2, 0);
  
  if(examples.length != colors.length)
    return;

  for (int set = 0; set < examples.length; set++) {
    if(examples[set] != null) {
      
      // The background activation is just for one set
      if(set == 0) {
        for (int i = 0; i < width / 2; i++) {
          for (int j = 0; j < height; j++) {
            // Changes the value of the input to match each possible point in the range of the screen
            inputUnitX.activationValue = map(i, 0, width / 2, minValue("x", examples[set]), maxValue("x", examples[set]));
            inputUnitY.activationValue = map(j, height, 0, minValue("y", examples[set]), maxValue("y", examples[set]));
                       
            Unit unit = findSelectedUnit() == null ? findOutputUnit() : findSelectedUnit();
            float cv = map(unit.activation(), 0, 1, 0, 255);
            color col = color(cv, cv, cv);
            set(i + width / 2, j, col);
          }
        }
      }
      
      strokeWeight(1);
      for (LearningExample example : examples[set]) {
        float cv = map(example.target, minValue("t", examples[set]), maxValue("t", examples[set]), 0, 255); 
        fill(cv, cv, cv);
        stroke(colors[set]);
        float ex = map(example.x, minValue("x", examples[set]), maxValue("x", examples[set]), 5, width / 2 - 5);
        float ey = map(example.y, minValue("y", examples[set]), maxValue("y", examples[set]), height - 5, 5);
        ellipse(ex, ey, 12, 12);
      }
    }
  }

  popMatrix();
  popStyle();
}

void drawGraph() {
  // Refresh the window
  pushStyle();
    fill(255);
    noStroke();
    rect(width / 2, 0, width, height);
  popStyle();  
  // Draw the learning graph
  if (graphToDraw == "data")
    dataGraph(activeSets, learningCols);
  else if (graphToDraw == "learningCurve")
    learningGraph(learningCols);
}

float logn(float x, float n) {
  return log(x) / log(n);
}

void draw() {
  // Refresh only the first half of the window, to address performance issues : drawing the graphs is expensive.
  fill(255);
  
  pushStyle();
    noStroke();
    rect(0, 0, width / 2, height);
  popStyle();
  
  textSize(25);
  strokeWeight(2);

  // Dividing line
  line(width / 2, 0, width / 2, height);

  if (batchLearning) {
    dataPoints = new ArrayList<PVector[]>();
    
    for(int i = 0; i < activeSets.length; i++)
      // +1 because we also want the initial error value, without any learning. 
      dataPoints.add(new PVector[nbEpochs + 1]);
    
    float squareErrorSum = 0;
    
    // We start at this value because it will be the first to be represented on the log scale (and will correspond to iteration 0, the first error value without learning)
    for (float i = 0; i <= nbEpochs; i++) {
      for(int sets = 0; sets < dataPoints.size(); sets++) {
        for(LearningExample example : activeSets[sets]) {
         // We collect information about activation and error for the current weight on each active set.
          inputUnitX.setActivationValue(example.x, activeSets[sets]);
          inputUnitY.setActivationValue(example.y, activeSets[sets]);
  
          float activation = findOutputUnit().activation();
          float error =  computeError(activation, example.target); //<>//
          
          squareErrorSum += pow(error, 2);
        
          // The weight learning only operates on the first set, and not in the first iteration which only checks the initial error //<>//
          if(sets == 0) {
            ArrayList<Float> gradients = new ArrayList<Float>();
            // The derivative of the error function with respect to the output
            gradients.add(-1.0);
            findOutputUnit().epoch(activation, gradients, error, null, 0);
            for(ArrayList<Unit> layer : layers) {
                for(Unit unit : layer) {
                  if(!unit.isInput)
                     unit.finishEpoch(error, 0);
                     unit.updateEdges();
                }
            }
            
          }
      }
        
        //if(i == 0 && actUnit.type == "RELU")
        //  println("error : " + squareErrorSum);
        
        // DECOMMENTER ICI POUR OFFLINE LEARNING
        //if(sets == 0)
        //  actUnit.finishEpoch();
        
        // The number whose multiples we want to appear on the scale. 
        float base = 10;

        //  Since we cannot compute log(0), if i = 0 we change its value (in a new variable in order to keep i only for the iteration process)
        float v = i == 0 ? 0.1 : i;
        
        float px = map(logn(v, base), logn(0.1, base), logn(nbEpochs, base), 0, width / 2);
        
        // Maximum granularity of the error
        if(squareErrorSum < 0.1)
          squareErrorSum = 0.1;
        
        float py = map(logn(squareErrorSum, base), logn(0.1, base), logn(1000, base), height - 10, 0);
        
        dataPoints.get(sets)[floor(i)] = new PVector(px, py);
        //println("x : " + dataPoints.get(sets)[i].x + " y : " + dataPoints.get(sets)[i].y);
        
        squareErrorSum = 0;
      }
    }

    batchLearning = false;
    drawGraph();
  }


  // Learning example by example animation
  if (stepLearning || presentExample) {
    LearningExample example = training.get(toLearnIndex);

    inputUnitX.setActivationValue(example.x, training);
    inputUnitY.setActivationValue(example.y, training);

    targetValue = str(example.target);
    
    for(Unit unit : layers.get(currentLayerIndex)) {
      // Animate
      unit.animate(t); 
    }
    

    t += 0.005;
    
    
    // Animation finished : Reset animations & prepare next example
    if (t > 1) {
      currentLayerIndex++;
      // Reset time
      t = 0;
    }
    
    if(currentLayerIndex == layers.size()) {
      currentLayerIndex = 1;
      
      // These operations are done on the output unit
      float activation = findOutputUnit().activation();
      findOutputUnit().shape.setStroke(color(202, 202, 202));
      // The string value to be inserted in the GUI
      activationValue = str(activation);
      float error = computeError(activation, example.target);
      // The error value to be inserted in the GUI
      errorValue = str(error);
      // Specifies the index of the next example to feed the network
      toLearnIndex = (toLearnIndex + 1) % training.size(); 
      
      // Make the unit learn.
      if (stepLearning) {
        // Different learning rates depending on the unit type for optimization.
        ArrayList<Float> gradients = new ArrayList<Float>();
        gradients.add(activation);
        findOutputUnit().learn(activation, gradients, selectLearningRate(findOutputUnit().activationType, "step"), error);
      }
      
      stepLearning = false;
      presentExample = false;
      drawGraph();
    }
  }

  strokeWeight(1);
  
  // The graphical activation unit, order of drawing important to avoid the overlap of edges on units
  for(int i = layers.size() - 1; i >= 0; i--)
    for(int j = layers.get(i).size() - 1; j >= 0; j--)
      layers.get(i).get(j).drawObject();
 

  float[] outvalues = null;
  int[] colors = null;

  if (activationValue != null) {
    outvalues = new float[2];
    outvalues[0] = float(activationValue);
    outvalues[1] = float(targetValue);
    colors = new int[2];
    colors[0] = color(0, 0, 0);
    colors[1] = color(0, 200, 50);
  }   

  // Draws the unit's activation function in a small size on top of it to visually indicate its type.
  // Different parameters depending on the unit type to optimize the visualisation
  for(ArrayList<Unit> layer : layers) {
    for(Unit unit : layer) {
      if(!unit.isInput)
        switch(unit.activationType) {
          case "SIGMOID":
            if(unit.isOutput)
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, outvalues, colors);
            else
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, null, null);
            break;
          case "TANH":
            if(unit.isOutput)          
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, outvalues, colors);
            else
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, null, null);
            break;
          case "SOFTPLUS":
            if(unit.isOutput)          
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, outvalues, colors);
            else
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, null, null);
            break;  
          case "RELU":
            if(unit.isOutput)          
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, outvalues, colors);
            else
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, null, null);  
            break;
          case "LRELU":
            if(unit.isOutput)          
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, outvalues, colors);
            else
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, null, null);  
            break;  
          case "ELU":
            if(unit.isOutput)          
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, outvalues, colors);
            else
              unit.drawActivationFunction(unit.pos.x, unit.pos.y, -10, 10, 100, 100, null, null);  
            break;    
          }
      }
  }
  
  Unit outunit = findOutputUnit();

  // -------------------- DRAW GUI ----------------------------------
  fill(255);
  rect(outunit.pos.x + 100, outunit.pos.y - 50 - 30, 110, 40);
  rect(outunit.pos.x + 100, outunit.pos.y + 50 + 30, 110, 40);
  rect(outunit.pos.x + 100, outunit.pos.y, 110, 40);
  fill(0);
  text("error", outunit.pos.x + 100, outunit.pos.y - 50 - 40);
  text("activation", outunit.pos.x + 100, outunit.pos.y - 7.5);
  text("target", outunit.pos.x + 100, outunit.pos.y + 50 + 20);
  textSize(16);  
  fill(0, 200, 20);
  if (targetValue != null)
    text(targetValue, outunit.pos.x + 100 + 10, outunit.pos.y + 50 + 20 + 35);
  fill(0, 0, 0);  
  if (activationValue != null)
    text(activationValue, outunit.pos.x + 100 + 10, outunit.pos.y + 25);  
  fill(200, 0, 20);
  if (errorValue != null)
    text(errorValue, outunit.pos.x + 100 + 10, outunit.pos.y - 50 - 40 + 35);
  //if (errorValue != null)
  fill(0);
  textSize(20);  
  String text = "Weight Init Mode : ";
  text(text, 10, 20);  
  fill(200);
  text(availableInits[abs(initsIterator) % availableInits.length], 10 + textWidth(text), 20);

  fill(0);
}

void mousePressed() {
  previousMouseClickX = mouseX;
  previousMouseClickY = mouseY;
}

void mouseClicked() {
  for(ArrayList<Unit> layer : layers) {
      for(Unit unit : layer) {
        if(unit.checkMouseCollision(mouseX, mouseY))
          unit.select(true);
        else
          unit.select(false);
      }
  }
  
  drawGraph();
}

void mouseDragged() {
  for(ArrayList<Unit> layer : layers) {
      for(Unit unit : layer) {
        if(unit.checkMouseCollision(mouseX, mouseY) && unit.selected)
          unit.move(mouseX - pmouseX, mouseY - pmouseY);
      }
  }
}

void keyPressed() {
  if (key == 'l') {
    // deactivate
    stepLearning = true;
    errorValue = null;
    activationValue = null;
  }

  // Reset weights
  else if (key == 'r') {
    for(ArrayList<Unit> layer : layers) {
      for(Unit unit : layer) {
        if(!unit.isInput)
          unit.resetWeights(availableInits[abs(initsIterator) % availableInits.length]);
      }
    }
    drawGraph();
    
  } else if (keyCode == RIGHT && findSelectedUnit() != null) {
    typeIterator++;
    findSelectedUnit().changeActivationType(availableTypes[abs(typeIterator) % availableTypes.length]);
    drawGraph();
  }else if (key == 'w') {
    initsIterator++;
  } else if (keyCode == LEFT && findSelectedUnit() != null) {
    typeIterator--;
    findSelectedUnit().changeActivationType(availableTypes[abs(typeIterator) % availableTypes.length]);
    drawGraph();
  } else if (keyCode == UP) {
    graphIterator++;
    graphToDraw = availableGraphs[abs(graphIterator) % availableGraphs.length];
    drawGraph();
  } else if (keyCode == DOWN) {
    graphIterator--;
    graphToDraw = availableGraphs[abs(graphIterator) % availableGraphs.length];
    drawGraph();
  } else if (key == 'p') {
    // deactivate
    presentExample = true;
    errorValue = null;
    activationValue = null;
  } else if (key == 'b') {
    // deactivate
    batchLearning = true;
  } else if (key == 's') {
    // Prints the current plot in an image
    PImage plot = get(width / 2 + 2, 0, width / 2, height);
    plot.save("plots/" + actUnit.activationType + "-" + "initMode=" + availableInits[abs(initsIterator) % availableInits.length] + "rate=" + selectLearningRate(actUnit.activationType, "batch") + availableGraphs[abs(graphIterator) % availableGraphs.length] + "-graph.png");
  } else if (key == 'v' && findSelectedUnit() != null) {
    // Prints the weight values
    findSelectedUnit().printUnit();
  } else if (key == 'n') {
    // Prints the weight values
      scaleNetworkWeights(2.0);
  }
  

  
}
