
// these are support function

function shuffle(array) {
  // this will shuffle all triallist
  // Fisher-Yates algorithm found on web
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  };
  return array;
};

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min)) + min;
};

function getDatetime(){
  // get date and time for record in data
  const datetime = new Date().toString();
  return datetime;
};

function getCurrentTrialNo(){
  // get the trial number from last trial, for reading data from triallist
  // use in main experiment loop except for fixation
  var curr_trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  return curr_trial_no;
};

function updateTrialNo(){
  // this is only used in the fixation function
  // align the trial no. with the main experiment loop
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  var trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  if ((typeof trial_no === "undefined") || (triallist[curr_triallist].length == trial_no + 1)){   // this will catch for the first trial
    trial_no = 0;
  } else {    // the rest will add one and align with the loop
    trial_no += 1;
  };
  return trial_no;
};

function getCurrentTriallist(){
  // get the triallist from last trial, for reading data from triallist
  // use in main experiment loop except for fixation
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  return curr_triallist;
};

function updateTriallist(){
  // this is only used in the fixation function
  // update the triallist number
  var trial_no = jsPsych.data.getLastTrialData().select('trial_no').values[0];
  var curr_triallist = jsPsych.data.getLastTrialData().select('curr_triallist').values[0];
  if (typeof curr_triallist === "undefined"){   // this will catch for the first trial
    curr_triallist = 0;
  } else if (triallist[curr_triallist].length === trial_no+1){
    curr_triallist += 1;
  } else {
    curr_triallist = getCurrentTriallist();
  };
  return curr_triallist;
};

function getCurrentRunBlockNo(trial_no, total_no_of_trials){
  // this compute run and block no to show in break screen
  // 4 runs 4 blocks of 32 trials
  var no_of_runs = 4;
  var no_of_blks = 4;
  var trial_in_run = no_of_blks * trials_to_break;
  var trial_in_blk = trials_to_break;

  var current_run = Math.floor((trial_no / trial_in_run) + 1);
  var current_blk = (trial_no - (Math.floor(trial_no / trial_in_run)) * trial_in_run)/trial_in_blk + 1;
  return [current_run, current_blk];
};

function checkAccuracy(data){
  var triallist_no = getCurrentTriallist();
  var trial = getCurrentTrialNo();
  const resp_array = ['bed', 'bridge', 'car', 'cat', 'elephant', 'flower', 'house', 'knife', 'phone', 'train'];
  data.response = resp_array[data.response];
  if (data.response === triallist[triallist_no][trial][2]){
    data.correct = 1;
  } else {
    data.correct = 0;
  };
  data.image_index = triallist[triallist_no][trial][1];
  data.category = triallist[triallist_no][trial][2];
  data.blur = triallist[triallist_no][trial][3];
  data.reps = triallist[triallist_no][trial][4];
};

var running_score = 0;
function getConfidence(data){
  var trial = getCurrentTrialNo();
  // add 1 to confidence to be 1 - 4
  if ((trial === 0) || (break_trial_array.includes(trial))){
    running_score = 0;
  };
  // handle running_score
  var is_it_correct = jsPsych.data.get().last(2).values()[0].correct;
  running_score += is_it_correct;
  data.running_score = running_score;
};

function getParameterByName(name, url) {
  if (!url) url = window.location.href;
  name = name.replace(/[\[\]]/g, "\\$&");
  console.log(url)
  var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)"),
      results = regex.exec(url);
  if (!results) return null;
  if (!results[2]) return '';
  return decodeURIComponent(results[2].replace(/\+/g, " "));
};

function hideCursor(){
  document.body.style.cursor = 'none';
};

function showCursor(){
  document.body.style.cursor = 'default';
};


function saveData(filename, filedata){
  $.ajax({
    type: 'post',
    cache: false,
    url: 'savedata.php',
    data: {filename: filename, filedata: filedata},
    success: function(data){
      window.location = "https://gatech-psych.sona-systems.com/webstudy_credit.aspx?experiment_id=1014&credit_token=55aac43119594f7faf12f780dadd577b&survey_code=" + getParameterByName("sona_id");
      console.log("Saved");
    },
  });
};
