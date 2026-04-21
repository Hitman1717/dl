w1,w2,bias=0.5,0.5,-1
def act(n):
    return 1 if n>=0 else 0
def train_network(inputs,outputs,learning_rate,epochs):
    global w1,w2,bias
    for e in range(epochs):
        total_error=0;
        for i in range(len(inputs)):
            a,b=inputs[i]
            result=act(w1*a+w2*b+bias)
            error=outputs[i]-result
            total_error+=abs(error)
            w1+=learning_rate*error*a
            w2+=learning_rate*error*b
            bias+=learning_rate*error
        if total_error==0:
            return
def NAND(a,b):
        return act(w1*a+w2*b+bias)
def xor(a,b):
    x=NAND(a,b)
    y=NAND(a,x)
    z=NAND(b,x)
    return NAND(y,z)    
inputs=[(0,0),(0,1),(1,0),(1,1)]
outputs=[1,1,1,0]
learning_rate=0.1
epochs=100
train_network(inputs, outputs, learning_rate, epochs)
for i in range(len(inputs)):
        a,b=inputs[i] 
        result=xor(a,b)
        print(f"inputs :{inputs[i]}  output:{result}")