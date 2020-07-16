let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new HGDataset();
var a=0, b=0, c=0, d=0, e=0;
let isPredicting = false;

async function loadMobilenet(){
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
    
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train(){
    dataset.ya = null;
    dataset.encodeLabels(5);

model = tf.sequential({
    layers: [
        tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
        tf.layers.dense({units: 140, activation: 'relu'}),
        tf.layers.dense({units: 5, activation: 'softmax'})
    ]
});

const optimizer = tf.train.adam(0.0001);

model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
let loss = 0;
model.fit(dataset.xs, dataset.ys, {
    epochs:10,
    callbacks: {
        onBatchEnd: async(batch, logs)=> {
            loss = logs.loss.toFixed(5);
            console.log('LOSS: ' + loss);
        }
    }
});
}

function handleButton(elem){
    switch(elem.id){
        case 'a':
            a++;
            document.getElementById('a').innerText = 'A Samples: ' + a;
            break;
        case 'b':
            b++;
            document.getElementById('b').innerText = 'B Samples: ' + b;
            break;
        case 'c':
            c++;
            document.getElementById('c').innerText = 'C Samples: ' + c;
            break;
        case 'd':
            d++;
            document.getElementById('d').innerText = 'D Samples: ' + d;
            break;
        case 'e':
            e++;
            document.getElementById('e').innerText = 'E Samples: ' + e;
            break;
    }
label = parseInt(elem.id);
const img = webcam.capture();
dataset.addExample(mobilenet.predict(img), label);
}

async function predict(){
    while(isPredicting){
        const predictedClass = tf.tidt(()=> {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
const classId = (await predictedClass.data())[0];
var predictionText = '';
switch(classId){
    case a:
        predictionText = 'It is A';
        break;
    case b:
        predictionText = 'It is B';
        break;
    case c:
        predictionText = 'It is C';
        break;
    case d:
        predictionText = 'It is D';
        break;
    case e:
        predictionText = 'It is E';
        break;
}
        document.getElementById('prediction').innerText = predictionText;
        predictedClass.dispose();
        await tf.nextFrame();
    }
}

function doTraining(){
    train();
    alert('Training Complete!');
}

function startPredicting(){
    isPredicting = true;
    predict();
    }
function stopPredicting(){
    isPredicting = false;
    predict();
    }
function saveModel(){
    model.save('downloads://my_model');
}
async function init(){
   //
    
    tf.setBackend('cpu');
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));    
}

init();