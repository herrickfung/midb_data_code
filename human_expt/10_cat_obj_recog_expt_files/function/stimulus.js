////////////////////////////////////////////////////////////////////////////////
// these functions creates stimulus

function createFixation(canvas) {
  var ctx = canvas.getContext('2d');
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(canvas.width/2-10, canvas.height/2);
  ctx.lineTo(canvas.width/2+10, canvas.height/2);
  ctx.moveTo(canvas.width/2, canvas.height/2-10);
  ctx.lineTo(canvas.width/2, canvas.height/2+10);
  ctx.stroke();
};


function getSrc(){
  var triallist_no = updateTriallist();
  var trial_no = updateTrialNo();
  var src = triallist[triallist_no][trial_no][0];
  full_src = `10_cat_obj_recog_expt_files/img/expt_stim/${src}`;
  return full_src;
};

function createImages(canvas){
  var ctx = canvas.getContext('2d');
  var testImg = new Image();

  var imageSize = 400;
  var triallist_no = getCurrentTriallist();
  var trial_no = getCurrentTrialNo();
  var src = triallist[triallist_no][trial_no][0];
  testImg.src = `10_cat_obj_recog_expt_files/img/expt_stim/${src}`
  testImg.onload = function(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(testImg, canvas.width/2 - imageSize/2, canvas.height/2-imageSize/2, imageSize, imageSize);
  };
};

function createBlank(canvas){
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,canvas.width,canvas.height);
};

function createRespButtonConfig(button_html) {
  const numButtons = 10;
  const cols = 5;       // 5 columns
  const rows = 2;       // 2 rows
  const buttonX = 150;
  const buttonY = 100;
  const gap = 10;       // Gap between buttons

  const startX = (600 - (cols * buttonX + (cols - 1) * gap)) / 2;
  const startY = 50 + (600 - (rows * buttonY + (rows - 1) * gap)) / 2;

  const buttonHTMLArray = [];
  for (let i = 0; i < numButtons; i++) {
    const row = Math.floor(i / cols);
    const col = i % cols;
    const left = startX + col * (buttonX + gap);
    const top = startY + row * (buttonY + gap);

    buttonHTMLArray.push(
      `<button class="jspsych-btn" style="position:absolute; left:${left}px; top:${top}px; width:${buttonX}px; height:${buttonY}px; font-size: 20px; font-weight: bold">%choice%</button>`
    );
  }
  return buttonHTMLArray;
};

function createConfButtonConfig(button_html){
  const numButtons = 11;
  const buttonX = 75;   // adjust width
  const buttonY = 200;   // adjust height
  const gap = 10;       // gap between buttons
  const totalWidth = numButtons * buttonX + (numButtons - 1) * gap;
  const startX = (600 - totalWidth) / 2;  // center horizontally
  const startY = 300;                     // vertical position

  const buttonHTMLArray = [];
  for (let i = 0; i < numButtons; i++) {
    const left = startX + i * (buttonX + gap);
    buttonHTMLArray.push(
      `<button class="jspsych-btn" style="position:absolute; left:${left}px; top:${startY}px; width:${buttonX}px; height:${buttonY}px; font-size: 20px; font-weight: bold">%choice%</button>`
    );
  }
  return buttonHTMLArray;
};

function createFeedbackScreen(){
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  var answer = triallist[triallist_no][trial][2]
  var is_it_correct = jsPsych.data.get().last(2).values()[0].correct;
  var feedback;
  if (is_it_correct === 1){
      feedback = '<p style="font-size:40px;color:#00ff00;"> CORRECT </p>'
    } else {
      feedback = `<p style="font-size:40px;color:#ff0000;padding:10px"> WRONG<br><br>Answer: ${answer} </p>`
      };
  return feedback;
};

function createStartPracticeScreen(){
  var instruction;
  var trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  if (typeof curr_triallist === "undefined"){   // this will catch for the first trial
      curr_triallist = 0;
    } else if (triallist[curr_triallist].length === trial_no+1){
        curr_triallist += 1;
      };

  switch (curr_triallist){
      case 0:
        instruction = start_prac_instruct_1;
        break;
      case 1:
        instruction = start_prac_instruct_2;
        break;
    };
 return instruction;
};

function createStartBreakScreen(){
  var current_trial_no = getCurrentTrialNo() + 1;
  var current_triallist = getCurrentTriallist();
  var total_no_of_trials = triallist[current_triallist].length;
  var running_score = jsPsych.data.get().last(1).values()[0].running_score;

  var break_text =
    '<p style="font-size:20px;color:#000000">' +
    'You got ' + running_score + ' out of ' + trials_to_break + ' trials correct in this block.' +
    '<br><br>You have completed ' + Math.round(current_trial_no/total_no_of_trials * 100) +
    '\% of the experiment.' +
     // '<br><br> RUN: ' + run_no + '<br><br> BLOCK: ' + block_no +
    '<br><br>Take a short break.' +
    '<br><br>When you are ready, click Continue to proceed.'

   return break_text;
};

function createEndBreakScreen(){
  var current_trial_no = getCurrentTrialNo() + 1;
  var current_triallist = getCurrentTriallist();
  var total_no_of_trials = triallist[current_triallist].length;
  var run_no = getCurrentRunBlockNo(current_trial_no, total_no_of_trials)[0];
  var block_no = getCurrentRunBlockNo(current_trial_no, total_no_of_trials)[1];

  var run_block_no_text =
    '<p style="font-size:32px;color:#000000">' +
    '<br><br> RUN: &nbsp ' + run_no + '/4 <br><br> BLOCK: ' + block_no + '/4 ';

  return run_block_no_text;
};



