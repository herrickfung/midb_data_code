// var noise_level = [8, 12];
var noise_level = [6, 10];
const categories = ['house', 'car', 'phone', 'bed', 'bridge', 'flower', 'train', 'cat', 'elephant', 'knife'];
const img_per_class_and_noise = 20;
const reps_per_img = 2;

function createTriallist(){
  var triallist = [];
  for (let rep = 0; rep < reps_per_img; rep++){
    var index = 0
    var sub_triallist = [];

    for (let cat = 0; cat < categories.length; cat++){
      for (let noise = 0; noise < noise_level.length; noise++){
        for (let k = 0; k < img_per_class_and_noise; k++){
            var trial = [`test/img_${index.toString().padStart(3, '0')}_test_${cat}_${noise_level[noise]}.png`, index, categories[cat], noise_level[noise], rep];
          sub_triallist.push(trial);
          index += 1;
        };
      };
    };
    shuffle(sub_triallist);
    triallist = triallist.concat(sub_triallist);
  };
  return triallist;
};

function createPracticeTriallist1(){
  var noise_level = [1];
  var triallist = [];
  index = 2;
  for (let cat = 0; cat < categories.length; cat++){
    for (let noise = 0; noise < 1; noise++){
      var trial = [`train/img_${index.toString().padStart(3, '0')}_train_${cat}_${noise_level[noise]}.png`, index, categories[cat], noise_level[noise]];
      triallist.push(trial)
      index += 10;
    };
  };
  shuffle(triallist);
  // triallist = triallist.slice(0, 1);
  return triallist;
};

function createPracticeTriallist2(){
  var triallist = [];
  index = 4;
  for (let cat = 0; cat < categories.length; cat++){
    for (let noise = 0; noise < noise_level.length; noise++){
      var trial = [`train/img_${index.toString().padStart(3, '0')}_train_${cat}_${noise_level[noise]}.png`, index, categories[cat], noise_level[noise]];
      triallist.push(trial)
      if (noise === 0){
        index += 3;
      } else {
        index += 7;
      };
    };
  };
  shuffle(triallist);
  // triallist = triallist.slice(0, 1);
  return triallist;
};

function breakTriallist(total_no_of_trials, trial_to_break){
  var no_of_breaks = Math.round(total_no_of_trials / trial_to_break);
  var break_trial_array = [];
  for (let i = 1; i < no_of_breaks; i++){
      break_trial_no = (total_no_of_trials / no_of_breaks * i);
      break_trial_no = Math.round(break_trial_no);
      break_trial_array.push(break_trial_no);
    };
  return break_trial_array;
};
