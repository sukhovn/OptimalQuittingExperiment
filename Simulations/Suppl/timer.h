#include <iostream>
#include <iomanip>
#include <sys/time.h>

class Print_Time{
private:
	int hours;
	int minutes;
	double seconds;
public:
	void update(double t){		
		hours = ((int) t) / 3600;
		t -= 3600.0 * hours;
		minutes = ((int) t) / 60;
		seconds = t - 60.0 * minutes;
	}
	Print_Time(double t){update(t);}

	friend std::ostream& operator<<(std::ostream& os, Print_Time prnt);
};

std::ostream& operator<<(std::ostream& os, Print_Time prnt){
	std::ios_base::fmtflags originalFlags = os.flags();

	if(prnt.hours > 0) os << prnt.hours << " hours ";
	if(prnt.hours > 0 || prnt.minutes > 0) os << prnt.minutes << " minutes ";
	os << std::fixed << std::setprecision(2);
	os << prnt.seconds << " seconds";
	os << std::scientific <<  std::setprecision(6);

	os.flags(originalFlags);
	return os;
}


class Timer{
private:
	// clock_t clck;
	struct timeval clck;
	double t_curr;
	double t_total;
	Print_Time pt_last_period;
	Print_Time pt_total;

public:
	const Print_Time &last_period;
	const Print_Time &total;

	Timer(void): t_curr(0.0), t_total(0.0), 
					pt_last_period(0.0), pt_total(0.0), 
					last_period(pt_last_period), total(pt_total) {gettimeofday(&clck, 0);}

	void record(void){
		struct timeval clck_end;
		gettimeofday(&clck_end, 0);
		long seconds = clck_end.tv_sec - clck.tv_sec;
	    long microseconds = clck_end.tv_usec - clck.tv_usec;
	    t_curr = seconds + microseconds*1e-6;
		clck = clck_end;

		t_total += t_curr;
		pt_last_period.update(t_curr);
		pt_total.update(t_total);
	}
};