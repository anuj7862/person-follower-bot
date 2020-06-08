#define left_motor_1  9

#define left_motor_2  10

#define right_motor_1  5

#define right_motor_2  6



void init_Motor(){

  pinMode(left_motor_1,OUTPUT);

  pinMode(left_motor_2,OUTPUT);

  pinMode(right_motor_1,OUTPUT);

  pinMode(right_motor_2,OUTPUT);

}




void setup() {

  // put your setup code here, to run once:

Serial.begin(9600);

init_Motor();

}



String incomingString = "";

int depth,depth_dir,angle,angle_dir;
int shift;
void loop() {

  if(Serial.available()){

    incomingString = Serial.readStringUntil('\n');
    
    angle_dir =  incomingString.substring(0,1).toInt();
    angle =  incomingString.substring(1,4).toInt();
    depth_dir = incomingString.substring(4,5).toInt();
    depth =  incomingString.substring(5,8).toInt();
    shift = angle/1.6;  ///**************
    Serial.println(incomingString);
    Serial.println(angle_dir);
    Serial.println(angle);
    Serial.println(depth_dir);
    Serial.println(depth);
    //depth = 100;
    motor(angle_dir,depth_dir,depth*5,shift);  
  }
   //analogWrite(left_motor_1,100);
   //digitalWrite(left_motor_2,LOW);
   //analogWrite(3,100);
   //digitalWrite(6,HIGH);   
}
int x;
void motor(int shift_dir,int depth_dir,int depth,int shift)
{
   Serial.print("depth    = ");
   Serial.println(depth);
         Serial.print("gaurav ");
       Serial.println(depth+shift);
    
  if(depth_dir == 1)
  {
    if(shift_dir == 1)
    {
        analogWrite( right_motor_1,depth);
      digitalWrite(right_motor_2,0);
      //if(depth < 14)
        analogWrite(left_motor_1,depth);
      //else  
       // analogWrite(left_motor_1,depth + shift);
      digitalWrite(left_motor_2, 0);
    }
  
    else
    {
      //if(depth < 14)
        analogWrite(right_motor_1,depth);
      //else  
        //analogWrite(right_motor_1,depth + shift);
      //analogWrite(right_motor_1,depth + shift);
      digitalWrite(right_motor_2,LOW);
      analogWrite( left_motor_1,depth);
      digitalWrite(left_motor_2,LOW);
    }
  }
  
  else
  {
    if(shift_dir == 1)
    {
      digitalWrite(left_motor_1,LOW);
      analogWrite( left_motor_2,depth);
      
      digitalWrite(right_motor_1,LOW);
      //if(depth < 14)
        //analogWrite(right_motor_2,1.5*shift);
      //else  
        analogWrite(right_motor_2,depth);
      //analogWrite(right_motor_2,depth  + shift);
    }
  
    else
    {
      digitalWrite(left_motor_1,LOW);
      //if(depth < 14)
       // analogWrite(left_motor_2, 1.5* shift);
      //else  
        analogWrite(left_motor_2,depth);
      //analogWrite(left_motor_2,depth +  shift);
      
      digitalWrite(right_motor_1,LOW);
      
      analogWrite( right_motor_2,depth);
    }
  }
}
