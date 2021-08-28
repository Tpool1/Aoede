function pause() {
    pywebview.api.pause()
}

function quit() {
    pywebview.api.quit()
}

function start() {
    pywebview.api.start()
}

function clear_user_data() {
    pywebview.api.clear_user_data()
}

function add_profile() {
    var name = document.getElementById("user-box").value;

    pywebview.api.add_profile(name);
}

function load_profiles() {
    pywebview.api.load_profiles().then(function(names) {
        document.querySelector('p').innerHTML = names;
    });
}
