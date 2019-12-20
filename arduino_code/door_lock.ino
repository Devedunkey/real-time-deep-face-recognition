#include <ESP8266WiFi.h>
#include <WiFiClient.h>
#include <ESP8266WebServer.h>
#include <ESP8266mDNS.h>

#ifndef STASSID
// wifi name
#define STASSID ""
// wifi password
#define STAPSK  ""
#endif

const char* ssid = STASSID;
const char* password = STAPSK;
// add userId
const char* user_name = "";
// add password
const char* user_pass = "";

ESP8266WebServer server(80);

const int led = 13;

void handleRoot() {
 if (!server.authenticate(user_name, user_pass)) {
    return server.requestAuthentication();
 }
 digitalWrite(led, HIGH);
 Serial.println("root request");
 server.send(200, "text/plain", "Door Open!");
 delay(3000);
 digitalWrite(led, LOW);
}

void handleNotFound() {
 digitalWrite(led, HIGH);
 String message = "File Not Found\n\n";
 message += "URI: ";
 message += server.uri();
 message += "\nMethod: ";
 message += (server.method() == HTTP_GET) ? "GET" : "POST";
 message += "\nArguments: ";
 message += server.args();
 message += "\n";
 for (uint8_t i = 0; i < server.args(); i++) {
   message += " " + server.argName(i) + ": " + server.arg(i) + "\n";
 }
 server.send(404, "text/plain", message);
 digitalWrite(led, LOW);
}

void setup(void) {
 pinMode(led, OUTPUT);
 digitalWrite(led, LOW);
 Serial.begin(115200);
 WiFi.mode(WIFI_STA);
 WiFi.begin(ssid, password);
 Serial.println("");

 // Wait for connection
 while (WiFi.status() != WL_CONNECTED) {
   delay(1000);
   Serial.print(".");
 }
 Serial.println("");
 Serial.print("Connected to ");
 Serial.println(ssid);
 Serial.print("IP address: ");
 Serial.println(WiFi.localIP());

 if (MDNS.begin("esp8266")) {
   Serial.println("MDNS responder started");
 }

 server.on("/", handleRoot);

 server.on("/inline", []() {
   Serial.println("inline request");
   server.send(200, "text/plain", "this works as well");
   digitalWrite(led, LOW);
 });

 server.onNotFound(handleNotFound);

 server.begin();
 Serial.println("HTTP server started");
}

void loop(void) {
 server.handleClient();
 MDNS.update();
}
