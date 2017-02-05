var RTCPeerConnection = null;
var getUserMedia = null;
var connectStreamToSrc = null;
var onMessage = null ;
var detectedBrowser = null;
var url = null;

if (navigator.getUserMedia) {
    // WebRTC standard
    RTCPeerConnection = RTCPeerConnection;
    getUserMedia = navigator.getUserMedia.bind(navigator);
    url = window.URL;
} else if (navigator.mozGetUserMedia) {
    // early Firefox
    detectedBrowser = "Firefox" ;
    RTCPeerConnection = mozRTCPeerConnection;
    RTCSessionDescription = mozRTCSessionDescription;
    RTCIceCandidate = mozRTCIceCandidate;
    getUserMedia = navigator.mozGetUserMedia.bind(navigator);
    url = window.URL;
}
else if (navigator.webkitGetUserMedia) {
    detectedBrowser = "Chrome" ;
    RTCPeerConnection = webkitRTCPeerConnection;
    getUserMedia = navigator.webkitGetUserMedia.bind(navigator);
    url = webkitURL;
} else {
    alert("WebRTC is not supported.");
}

var server = {
    "iceServers": [
    {url:"stun:stun.l.google.com:19302"}]
};
