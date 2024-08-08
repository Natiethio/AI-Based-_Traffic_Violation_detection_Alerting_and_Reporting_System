const int redPin = 4;    // Pin connected to the red LED
const int yellowPin = 3; // Pin connected to the yellow LED
const int greenPin = 2;  // Pin connected to the green LED

void setup() {
  // Set the pin modes for the LEDs
  pinMode(redPin, OUTPUT);
  pinMode(yellowPin, OUTPUT);
  pinMode(greenPin, OUTPUT);

  // Initialize serial communication
  Serial.begin(9600);
}

void loop() {

red();
yellow();
green();
// yellow();
  
}


void red(){
  
  // Turn on the red LED for 15 seconds
  digitalWrite(redPin, HIGH);
  sendStatus("red");
  delay(15000); // 15 seconds
  digitalWrite(redPin, LOW);
  }

void yellow(){
  
    // Turn on the yellow LED for 3 seconds
  digitalWrite(yellowPin, HIGH);
  sendStatus("yellow");
  delay(3000); // 3 seconds
  digitalWrite(yellowPin, LOW);
  }

void green(){
    // Turn on the green LED for 15 seconds
  digitalWrite(greenPin, HIGH);
  sendStatus("green");
  delay(15000); // 15 seconds
  digitalWrite(greenPin, LOW);
  }
void sendStatus(String status) {
  Serial.println(status);
}
